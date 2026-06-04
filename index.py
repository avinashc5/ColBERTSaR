import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, List
from tqdm.auto import tqdm

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import TensorDataset
from transformers import Trainer
from transformers.utils.logging import get_logger
from transformers.trainer_utils import is_main_process

from pylate.models import ColBERT

from args import Arguments, parse_args
from module import CentroidModel, CentroidRestartCallback, batch_encode_docs_from_generator, encode_queries
from sparse_index import ColBERTSaRIndexer

logger = get_logger(__name__)

@dataclass
class IndexingArguments(Arguments):
    with_weighted_assignments: bool = False

    # sampling options
    assumed_doclen: Optional[int] = 120

    # training queries
    training_queries: Optional[str] = None
    distributed_query_encoding: bool = False
    max_n_queries: int = 8192

    reinit_centroids_at_end_of_epoch: bool = False

    # training doc embeddings
    training_embedding_samples: Optional[List[str]] = None
    init_centroids: Optional[str] = None
    n_sample_embeddings: Optional[int] = None
    n_centroids: Optional[int] = None
    overwrite_encoded_samples: Optional[bool] = False

    # training objectives
    with_kmeans_objective: bool = False
    max_difference_objective: bool = False
    entropy_regularization: float = 0.0

    # soft assignment with temperature annealing
    use_soft_assignment: bool = False
    initial_temperature: float = 1.0
    final_temperature: float = 0.01
    temperature_anneal_steps: Optional[int] = None  # defaults to max_steps if None

    # indexing
    chunk_size: Optional[int] = 50000
    n_way_merging: Optional[int] = 8
    n_merging_worker: Optional[int] = 8

    # misc
    clean_up_samples: Optional[bool] = False

    # forced user actions
    retrain_centroids: Optional[bool] = False
    do_sampling_only: Optional[bool] = False
    do_merging_only: Optional[bool] = False

    do_sampling: Optional[bool] = True
    do_training: Optional[bool] = True
    do_encoding: Optional[bool] = True
    do_merging: Optional[bool] = True

    def has_centroid_file(self) -> bool:
        assert isinstance(self.centroids, str)
        return Path(self.centroids).exists()

    def has_sample_files(self) -> bool:
        return self.training_embedding_samples is not None and all(Path(f).exists() for f in self.training_embedding_samples)

    def __post_init__(self):
        super().__post_init__()
        assert isinstance(self.output_dir, (str, Path))
        output_dir = Path(self.output_dir)

        if self.n_centroids is None:
            if self.centroids is not None and Path(self.centroids).exists():
                loaded_centroids = torch.load(self.centroids, map_location='cpu')
                self.n_centroids = loaded_centroids.size(0)
                logger.info(f"Set n_centroids to {self.n_centroids} from loaded centroids file {self.centroids}")
            elif self.init_centroids is not None and Path(self.init_centroids).exists():
                loaded_centroids = torch.load(self.init_centroids, map_location='cpu')
                self.n_centroids = loaded_centroids.size(0)
                logger.info(f"Set n_centroids to {self.n_centroids} from init centroids file {self.init_centroids}")
            else:
                raise ValueError("n_centroids is not set and cannot be inferred from centroids/init_centroids file")

        if self.training_embedding_samples is None:
            existing_samples = list(output_dir.glob("samples.*.pt"))
            if len(existing_samples) > 0:
                self.training_embedding_samples = list(map(str, existing_samples))
            else:
                self.training_embedding_samples = [
                    str(output_dir / f"samples.{i}.pt") for i in range(self.world_size)
                ]

        if self.training_queries is None:
            logger.warning("Training queries not set, falling back to in-batch mode")
            self.training_queries = 'in-batch'

        assert sum([self.retrain_centroids, self.do_sampling_only, self.do_merging_only, self.resume]) <= 1, \
                "should only select at most one one of resume, retrain_centroids, do_sampling_only and do_merging_only"

        metadata: dict[str, Any] | None = None
        if (output_dir / "metadata.json").exists():
            metadata = json.loads((output_dir / "metadata.json").read_text())

        if self.centroids is None:
            if metadata is not None and "centroids" in metadata:
                self.centroids = metadata['centroids']
            else:
                self.centroids = str((output_dir / "centroids.pt").absolute())


        n_encoded_shards = None
        if metadata is not None and 'total_shards' in metadata:
            n_encoded_shards = len(list(output_dir.glob("centroid_shard_*.pkl")))

        if self.do_sampling_only:
            self.do_sampling, self.do_training, self.do_encoding, self.do_merging = (True, False, False, False)

        if self.do_merging_only and metadata is not None:
            assert self.has_centroid_file()
            if 'total_shards' in metadata:
                assert n_encoded_shards == metadata['total_shards']
            else:
                logger.warning("Don't have total_shards in metadata, not checking if all shards are finished encoding")
            self.do_sampling, self.do_training, self.do_encoding, self.do_merging = (False, False, False, True)

        if self.retrain_centroids:
            self.do_training, self.do_encoding, self.do_merging = (True, True, True)

        if self.do_sampling and self.has_sample_files():
            if self.overwrite_encoded_samples:
                logger.warning("Found sample files but force overwriting them")
            else:
                logger.info("Found sample files, will reuse them")
                self.do_sampling = False

        if sum([self.retrain_centroids, self.do_sampling_only, self.do_merging_only]) == 1:
            return

        if metadata is not None and metadata['nnz'] > 0 and metadata['n_docs'] > 0 and len(list(output_dir.glob("*.npy"))) == 4 and self.has_centroid_file():
            self.do_sampling, self.do_training, self.do_encoding, self.do_merging = (False, False, False, False)
            return

        self.do_training = not self.has_centroid_file()
        self.do_sampling = self.do_training and not self.has_sample_files()
        self.do_encoding = metadata is None or n_encoded_shards != metadata.get('total_shards', -1)
        self.do_merging = True


    def get_training_queries(self):
        queries = sorted(Arguments.read_queries(self.training_queries).values())

        if self.max_n_queries is not None and len(queries) > self.max_n_queries:
            return queries[::len(queries)//self.max_n_queries]

        return queries


def sample(args: IndexingArguments, output_dir: Path):
    n_passages = args.get_collection_len()
    n_samples = args.n_sample_embeddings

    if n_samples is None:
        # default sampling strategy from PLAID
        n_samples = 16 * np.sqrt(args.assumed_doclen * n_passages)
        n_samples = min(1 + int(n_samples), n_passages)

    sample_pids = None
    if is_main_process(args.local_rank):
        logger.info(f"Sampling {n_samples} from {n_passages} passages.")
        sample_pids = sorted(np.random.choice(n_passages, size=n_samples, replace=False).tolist())

        with open(output_dir / "sampled_pids.txt", "w") as f:
            for pid in sample_pids:
                f.write(f"{pid}\n")

    if dist.is_initialized():
        gathering = [None for _ in range(args.world_size)]
        dist.all_gather_object(gathering, sample_pids)
        sample_pids = gathering[0]

    sample_pids = set(sample_pids)

    colbert_checkpoint = ColBERT(model_name_or_path=args.colbert_checkpoint, model_kwargs={
        'dtype': torch.float16 if args.fp16 else torch.float32
    }, tokenizer_kwargs={"use_fast": True}).eval().to(args.device)

    coll_generator, _ = args.get_collection_generator(need_mapper=False)
    batch_D_generator = batch_encode_docs_from_generator(
        (doc for i, doc in enumerate(coll_generator) if i in sample_pids),
        colbert_checkpoint,
        keep_docs_in_batch=False,
        batch_size=args.per_device_eval_batch_size,
        n_docs=len(sample_pids),
        desc="Encoding sample passages",
        local_rank=args.local_rank,
        world_size=args.world_size
    )

    encoded_vectors = torch.concat([
        D.half().cpu()
        for pids, passages, D in batch_D_generator
    ])

    torch.save(encoded_vectors, output_dir / f"samples.{args.local_rank}.pt")

    if dist.is_initialized():
        dist.barrier()


def _encode_queries(args: IndexingArguments, queries: list[str]):
    colbert_checkpoint = ColBERT(args.colbert_checkpoint).eval().to(args.device)

    if args.distributed_query_encoding:
        queries = queries[args.local_rank::args.world_size]

    Qs = torch.concat(encode_queries(
        queries[args.local_rank::args.world_size],
        colbert_checkpoint,
        args.per_device_eval_batch_size,
        show_progress_bar=is_main_process(args.local_rank)
    )).float()

    if args.distributed_query_encoding:
        gather_Qs = [ torch.zeros_like(Qs) for _ in range(args.world_size) ]
        dist.all_gather(gather_Qs, Qs)
        Qs = torch.concat(gather_Qs)

    return Qs.contiguous().cuda()



def train(args: IndexingArguments, output_dir: Path):
    assert Path(args.centroids).is_relative_to(output_dir.absolute()), \
        f"Specified --centroids {args.centroids} is not under the output_dir {output_dir}"

    args.ddp_find_unused_parameters = False

    training_Qs = None
    if args.training_queries != 'in-batch':
        training_Qs = _encode_queries(args, args.get_training_queries())

    torch.cuda.empty_cache()

    samples = torch.concat([
        torch.load(fn, map_location="cpu") for fn in tqdm(
            args.training_embedding_samples, desc="Loading embedding samples",
            disable=not is_main_process(args.local_rank)
        )
    ])

    # shuffle so that, for MLIR, a single batch can contain embeddings from different languages
    samples = samples[torch.randperm(samples.shape[0])]
    if args.fp16:
        samples = samples.half()

    if args.init_centroids is None:
        assert args.n_centroids is not None, "Must specify number of centroids if no initial centroids provided"
        random_indices = torch.randperm(samples.size(0))[:args.n_centroids]
        loaded_centroids = samples[random_indices].float().clone()
    else:
        loaded_centroids = torch.load(args.init_centroids, map_location='cuda').float()

    model = CentroidModel(
        init_centroids=loaded_centroids,
        with_weighted_assignments=args.with_weighted_assignments,
        training_Qs=training_Qs,
        with_kmeans_objective=args.with_kmeans_objective,
        max_difference_objective=args.max_difference_objective,
        entropy_regularization=args.entropy_regularization,
        use_soft_assignment=args.use_soft_assignment,
        initial_temperature=args.initial_temperature,
        final_temperature=args.final_temperature,
        temperature_anneal_steps=args.temperature_anneal_steps or args.max_steps
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=TensorDataset(samples),
        data_collator=lambda x: {"batch_dvecs": torch.stack([e[0] for e in x])}
    )
    if args.reinit_centroids_at_end_of_epoch:
        trainer.add_callback(CentroidRestartCallback(trainer=trainer, sample_query_size=args.max_n_queries))

    trainer.train(resume_from_checkpoint=args.resume and len(list(output_dir.glob("checkpoint-*"))) > 0)

    if is_main_process(args.local_rank):
        torch.save(model.centroids.detach().cpu(), args.centroids)
        logger.info(f"Saved trained centroids at {args.centroids}")

        if args.clean_up_samples:
            logger.info("Clean up samples")
            for fn in output_dir.glob("samples.*.pt"):
                fn.unlink()

    if dist.is_initialized():
        dist.barrier()


def encode(indexer: ColBERTSaRIndexer, args: IndexingArguments, output_dir: Path):
    colbert_checkpoint = ColBERT(model_name_or_path=args.colbert_checkpoint, model_kwargs={
        'dtype': torch.float16 if args.fp16 else torch.float32
    }, tokenizer_kwargs={"use_fast": True}).eval().to(args.device)

    centroids: torch.Tensor = torch.load(args.centroids, map_location=args.device)
    if args.fp16:
        centroids = centroids.half()
    
    centroid_model = CentroidModel(
        centroids,
        with_weighted_assignments=args.with_weighted_assignments
    ).eval()

    batch_D_generator = batch_encode_docs_from_generator(
        args.get_collection_generator(need_mapper=False, force_pid_to_int=True)[0],
        colbert_checkpoint,
        keep_docs_in_batch=True,
        batch_size=args.per_device_eval_batch_size,
        n_docs=args.get_collection_len(),
        desc="Encoding passages",
        local_rank=args.local_rank,
        world_size=args.world_size
    )

    for pids, passages, D in batch_D_generator:
        if args.fp16:
            D = D.half()
        assignments, weights = centroid_model(D, is_training=False)
        batched_doc_terms = assignments.tolist()
        if args.with_weighted_assignments:
            batched_weights = weights.tolist()
            assert len(batched_weights) == len(batched_doc_terms)
        else:
            batched_weights = [None]*len(batched_doc_terms)

        for pid, cidx, pws in zip(pids, batched_doc_terms, batched_weights):
            if pws is None:
                sparse_vec = {cid: 1. for cid in cidx}
            else:
                sparse_vec = {}
                for cid, w in zip(cidx, pws):
                    sparse_vec[cid] = max(w, sparse_vec.get(cid, -1))
            indexer.add_document(pid, sparse_vec)

    indexer.flush()

    if dist.is_initialized():
        dist.barrier()

    if is_main_process(args.local_rank):
        indexer.update_total_n_shards()


def main(args: IndexingArguments):

    output_dir = Path(args.output_dir)

    logger.info(
        f"Indexing pipeline: sampling={args.do_sampling}, training={args.do_training}, "
        f"encoding={args.do_encoding}, merging={args.do_merging} -> {output_dir}"
    )
    logger.info(
        f"Model: {args.colbert_checkpoint} | n_centroids: {args.n_centroids} | "
        f"training_queries: {args.training_queries} | fp16: {args.fp16}"
    )

    if args.do_sampling:
        logger.info("[1/4] Sampling passages for centroid training")
        sample(args, output_dir)
    else:
        logger.info("[1/4] Skipping sampling (samples already exist on disk)")

    if args.do_training:
        logger.info(f"[2/4] Training {args.n_centroids} centroids for {args.max_steps} steps")
        train(args, output_dir)
    else:
        logger.info(f"[2/4] Skipping centroid training (using existing centroids at {args.centroids})")

    indexer = ColBERTSaRIndexer(
        args.output_dir,
        n_centroids=args.n_centroids,
        write_freq=args.chunk_size,
        local_rank=args.local_rank,
        metadata={
            'colbert_checkpoint': args.colbert_checkpoint,
            'centroids': args.centroids,
            'collection': args.collection,
            'passage_mapping': args.passage_mapping
        }
    )

    if args.do_encoding:
        logger.info("[3/4] Encoding the document collection with the trained centroids")
        encode(indexer, args, output_dir)
    else:
        logger.info("[3/4] Skipping encoding (centroid shards already on disk)")

    if args.do_merging:
        logger.info("[4/4] Merging shards into forward and inverted indexes")
        indexer.create_index(
            args.n_way_merging,
            args.n_merging_worker
        )
        if is_main_process(args.local_rank):
            logger.info(
                f"Index ready at {output_dir} -- "
                f"n_docs={indexer.n_docs}, n_centroids={indexer.n_centroids}, nnz={indexer.nnz}"
            )
    else:
        logger.info("[4/4] Skipping merging (index already built)")


if __name__ == '__main__':
    main(parse_args(IndexingArguments))
