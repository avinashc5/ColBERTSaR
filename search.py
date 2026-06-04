from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Optional, List

import torch
import torch.distributed as dist
import numpy as np
import scipy.sparse as sp
from transformers.utils.logging import get_logger
from transformers.trainer_utils import is_main_process
from pylate.models import ColBERT
from tqdm.auto import tqdm

from args import Arguments, parse_args
from utils import maxp, write_trec_run, read_trec_run
from module import encode_queries
from sparse_index import _load_mmap_spm

logger = get_logger(__name__)


def _same_numpy_array(a: np.ndarray, b: np.ndarray):
    return a.__array_interface__['data'][0] == b.__array_interface__['data'][0]


@dataclass
class SearchArguments(Arguments):
    queries: Optional[str] = None
    qrels: Optional[str] = None
    metrics: List[str] = field(default_factory=lambda :['nDCG@10', 'nDCG@20', 'P@10', 'R@1000'])
    search_output: Optional[str] = None

    topk: int = 1000
    nprobe: Optional[int] = 1
    batch_search: Optional[bool] = False

    use_forward_index: Optional[bool] = True
    load_full_index_to_memory: Optional[bool] = False

    def get_queries(self):
        return Arguments.read_queries(self.queries)


class ColBERTSaRSearcher:
    def __init__(
        self,
        index_dir: str,
        colbert_checkpoint: ColBERT = None,
        centroids: str = None,
        with_forward_index: bool = True,
        use_mmap: bool = True,
        device: torch.device = torch.device('cpu'),
    ):
        self.index_dir = Path(index_dir)
        with open(self.index_dir / "metadata.json") as f:
            self.metadata: dict[str, Any] = json.load(f)

        self.colbert_checkpoint = ColBERT(
            model_name_or_path=colbert_checkpoint or self.metadata['colbert_checkpoint'],
            tokenizer_kwargs={"use_fast": True}
        ).eval().to(device)

        self.centroids: torch.Tensor = torch.load(
            centroids or self.metadata['centroids'],
            map_location=device
        )

        self.inv_index: sp.csr_matrix = None
        self.for_index: sp.csr_matrix = None
        self.load_sparse_index(with_forward_index, use_mmap)

    @property
    def n_centroids(self):
        return self.metadata['n_centroids']

    @property
    def n_docs(self):
        return self.metadata['n_docs']

    @property
    def nnz(self):
        return self.metadata['nnz']

    def load_sparse_index(self, with_forward_index=True, use_mmap=True):
        logger.info(f"Loading index from {self.index_dir}")

        self.inv_index = _load_mmap_spm(
            str(self.index_dir / "inverted"),
            nnz=self.nnz,
            shape=(self.n_centroids, self.n_docs),
            use_mmap=use_mmap
        )
        logger.info("Loaded inverted index")

        if with_forward_index:
            self.for_index  = _load_mmap_spm(
                str(self.index_dir / "forward"),
                nnz=self.nnz,
                shape=(self.n_docs, self.n_centroids),
                use_mmap=use_mmap
            )
            logger.info("Loaded forward index")
        else:
            logger.info("Not loading forward index")

    def search_code(self, qcode: list[dict[int, float]], cscore: torch.Tensor = None, topk: int = 1000, bsize: int = 32):
        cidxs, scores = [], []
        for term in qcode:
            ks, vs = list(zip(*term.items()))
            cidxs += ks
            scores += vs

        nprobs = len(qcode[0])

        small_spm: sp.csr_matrix = self.inv_index[cidxs]
        # unweighted indexes store the same array for data and indices; weighted indexes have a separate data array
        if _same_numpy_array(self.inv_index.data, self.inv_index.indices):
            small_spm.data = np.repeat(scores, small_spm.indptr[1:] - small_spm.indptr[:-1])
        else:
            small_spm.data = small_spm.data * np.repeat(scores, small_spm.indptr[1:] - small_spm.indptr[:-1])

        scores = np.array(sp.vstack([
            small_spm[i*nprobs:(i+1)*nprobs].max(0) for i in range(len(qcode))
        ]).sum(axis=0)).ravel()

        topk_indices = scores.argsort()[-topk:][::-1]
        first_stage_results = dict(zip(topk_indices.tolist(), scores[topk_indices].tolist()))

        if self.for_index is None:
            return first_stage_results

        pids = sorted(first_stage_results.keys())
        centroid_ids = torch.from_numpy(np.concatenate([
            self.for_index.indices[ self.for_index.indptr[pid]: self.for_index.indptr[pid+1] ]
            for pid in pids
        ])).to(cscore.device)
        doc_ptr = torch.tensor([0] + [
            self.for_index.indptr[pid+1] - self.for_index.indptr[pid] for pid in pids
        ]).to(cscore.device).cumsum(dim=0)

        sims = cscore[:, centroid_ids]

        n_docs = len(pids)
        scores_tensor = torch.zeros(n_docs, device=cscore.device)

        for i in range(n_docs):
            start, end = doc_ptr[i], doc_ptr[i+1]
            scores_tensor[i] = sims[:, start:end].max(dim=1).values.sum()

        return dict(zip(pids, scores_tensor.tolist()))

    def embed_queries(self, queries: list[str], nprobe: int = 4, bsize: int = 32, with_tqdm: bool = True):
        Q = torch.stack(encode_queries(queries, self.colbert_checkpoint, batch_size=bsize, show_progress_bar=with_tqdm))
        cscores = (Q.to(dtype=self.centroids.dtype) @ self.centroids.T)
        val, idx = cscores.max(dim=-1, keepdim=True) if nprobe == 1 else torch.topk(cscores, k=nprobe, dim=-1)
        return [
            ([ dict(zip(ii,vv)) for ii, vv in zip(i, v) ], cscores[qi])
            for qi, (i, v) in enumerate(zip(idx.tolist(), val.tolist()))
        ]

    def search_query(self, query: str, nprobe: int = 4, topk: int = 1000, bsize: int = 32):
        return self.batch_search_queries([query], nprobe, topk, bsize, with_tqdm=False)[0]

    def batch_search_queries(self, queries: list[str], nprobe: int = 4, topk: int = 1000, bsize: int = 32, with_tqdm: bool = True):
        return [
            self.search_code(qcode, cscore, topk, bsize)
            for qcode, cscore in tqdm(
                self.embed_queries(queries, nprobe, with_tqdm=with_tqdm),
                desc='Search queries',
                dynamic_ncols=True,
                disable=not with_tqdm
            )
        ]


def main(args: SearchArguments):
    logger.info(
        f"Loading index from {args.index_dir} (mmap={not args.load_full_index_to_memory}, "
        f"forward_index={args.use_forward_index})"
    )
    searcher = ColBERTSaRSearcher(
        args.index_dir,
        colbert_checkpoint=args.colbert_checkpoint,
        centroids=args.centroids,
        with_forward_index=args.use_forward_index,
        use_mmap=not args.load_full_index_to_memory,
        device=args.device if str(args.device) != "cpu:0" else "cpu"
    )
    logger.info(
        f"Index ready -- n_docs={searcher.n_docs}, n_centroids={searcher.n_centroids}, nnz={searcher.nnz}"
    )

    if args.passage_mapping is None:
        args.passage_mapping = searcher.metadata.get("passage_mapping", None)

    pid_mapping = args.get_pid_mapping(force_pid_to_int=True, with_tqdm=is_main_process(args.local_rank))

    queries = args.get_queries()
    sorted_qids = sorted(queries.keys())
    logger.info(f"Loaded {len(sorted_qids)} queries from {args.queries}")

    if dist.is_initialized():
        sorted_qids = sorted_qids[args.local_rank::args.world_size]
        logger.info(f"Rank {args.local_rank}/{args.world_size}: handling {len(sorted_qids)} queries")

    sorted_queries = [ queries[qid] for qid in sorted_qids ]

    logger.info(f"Searching with nprobe={args.nprobe}, topk={args.topk}")

    if args.batch_search:
        all_scores = dict(zip(
            sorted_qids,
            searcher.batch_search_queries(
                sorted_queries,
                nprobe=args.nprobe,
                topk=args.topk,
                bsize=args.per_device_eval_batch_size
            )
        ))
    else:
        all_scores = {
            qid: searcher.search_query(query, nprobe=args.nprobe, topk=args.topk)
            for qid, query in zip(tqdm(sorted_qids, desc="Search queries", dynamic_ncols=True, disable=not is_main_process(args.local_rank)), sorted_queries)
        }

    all_scores = {
        qid: maxp(scores, pid_mapping, limit=100) for qid, scores in all_scores.items()
    }

    if dist.is_initialized():
        if args.search_output is not None:
            logger.info("Writing separate trec files to be safe")
            write_trec_run(all_scores, args.search_output + f".{args.local_rank}", run_id="colbertsar", topk=args.topk)
            dist.barrier()

            if not is_main_process(args.local_rank):
                return

            all_scores = {}
            for i in range(args.world_size):
                all_scores.update( read_trec_run(args.search_output + f".{i}") )

        else:
            logger.info(f"Gathering from {args.world_size} workers")
            gather_buffer = [None for _ in range(args.world_size)]
            dist.all_gather_object(gather_buffer, all_scores)
            for scores in gather_buffer:
                all_scores.update(scores)
            if not is_main_process(args.local_rank):
                return

    if args.search_output is not None:
        write_trec_run(all_scores, args.search_output, run_id="colbertsar", topk=args.topk)
        logger.info(f"Wrote run file to {args.search_output}")
        if dist.is_initialized():
            logger.info("Cleaning up per-worker temp files")
            for fn in Path().glob(args.search_output + ".*"):
                fn.unlink()

    if args.qrels is not None:
        import ir_measures as irms
        metrics = [ irms.parse_measure(m) for m in args.metrics ]
        if args.qrels.startswith('irds:'):
            import ir_datasets as irds
            qrels = irds.load(args.qrels.replace("irds:", "")).qrels
        else:
            qrels = list(irms.read_trec_qrels(args.qrels))
        logger.info(f"Evaluating against {args.qrels} with metrics {args.metrics}")
        results = irms.calc_aggregate(metrics, qrels, all_scores)
        for measure, score in sorted(results.items(), key=lambda x: str(x[0])):
            logger.info(f"{measure}: {score:.4f}")


if __name__ == '__main__':
    main(parse_args(SearchArguments, check_output_dir=False, write_args=False, print_args=False))
