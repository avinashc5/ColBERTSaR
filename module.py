import os

import torch
import torch.nn as nn
import torch.distributed as dist
from pylate.models import ColBERT
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.utils.logging import get_logger
from transformers.trainer_utils import is_main_process

from tqdm import tqdm

from utils import split_by_rank, batching

logger = get_logger(__name__)

def batch_encode_docs_from_generator(
        doc_generator,
        colbert_checkpoint: ColBERT,
        keep_docs_in_batch: bool,
        batch_size: int,
        n_docs: int,
        desc: str = "Encoding passages",
        local_rank: int = int(os.environ.get("LOCAL_RANK", 0)),
        world_size: int = int(os.environ.get("WORLD_SIZE", 1))
    ):
    reader = tqdm(
        split_by_rank(
            batching(doc_generator, bs=batch_size),
            local_rank, world_size
        ),
        total=n_docs//world_size//batch_size+1,
        disable=not is_main_process(local_rank), dynamic_ncols=True,
        desc=desc
    )

    with torch.inference_mode():
        colbert_checkpoint.eval()

        for batch in reader:
            pids, passages = list(zip(*batch))
            D: torch.Tensor = torch.concat(colbert_checkpoint.encode(
                passages,
                is_query=False, convert_to_numpy=False, convert_to_tensor=True, padding=keep_docs_in_batch,
                batch_size=batch_size, show_progress_bar=False
            ), dim=0)  # ty:ignore[no-matching-overload]

            yield pids, passages, D

def encode_queries(
        queries: list[str],
        colbert_checkpoint: ColBERT,
        batch_size: int = 32,
        show_progress_bar: bool = False
    ):
    with torch.inference_mode():
        return colbert_checkpoint.encode(
            queries,
            is_query=True,
            convert_to_numpy=False,
            convert_to_tensor=True,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar
        )

class CentroidRestartCallback(TrainerCallback):
    def __init__(self, trainer: Trainer, sample_query_size: int = 10000):
        self.trainer: Trainer = trainer
        self.train_dataset: torch.utils.data.TensorDataset = trainer.train_dataset

        model: CentroidModel = trainer.accelerator.unwrap_model(self.trainer.model)
        if model.training_Qs is None:
            sample_every = max(1, len(self.train_dataset) // sample_query_size)
            self.sample_queries = self.train_dataset.tensors[0][::sample_every]
        else:
            self.sample_queries = model.training_Qs

    def on_epoch_end(self, args: TrainingArguments, state, control, **kwargs):
        model: CentroidModel = kwargs['model']
        reader = tqdm(
            split_by_rank(
                batching(enumerate(self.train_dataset.tensors[0]), bs=args.per_device_train_batch_size),
                args.local_rank, args.world_size
            ),
            total=len(self.train_dataset)//args.world_size//args.per_device_train_batch_size+1,
            disable=not is_main_process(args.local_rank),
            dynamic_ncols=True,
            desc="Evaluating centroids for re-initialization"
        )

        queries = self.sample_queries.to(model.centroids.device)

        # 1. iterate through all vectors in training set, store touched centroids and keep track of prediction error of each example
        # 2. identify centroids that were never touched
        # 3. re-initialize those centroids with training vectors that are predicted with highest error

        touched_centroids = set()
        vectors_and_losses = []
        with torch.inference_mode():
            for batch_idx, inp in enumerate(reader):
                vecs = torch.stack([ vec for i, vec in inp]).to(model.centroids.device)
                assert all(vecs[0] == self.train_dataset.tensors[0][inp[0][0]].to(model.centroids.device))

                loss, assignments = model.forward(batch_dvecs=vecs, is_training=False, queries=queries)
                touched_centroids.update(assignments.view(-1).cpu().tolist())
                vectors_and_losses.extend(zip((i for i, vec in inp), loss.cpu().tolist()))

        if dist.is_initialized():
            all_touched_centroids = [set() for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_touched_centroids, touched_centroids)
            touched_centroids = set().union(*all_touched_centroids)

        untouched_centroids = sorted(set(range(model.centroids.shape[0])) - touched_centroids)
        logger.info(f"Re-initializing {len(untouched_centroids)} untouched centroids...")

        # local sort
        vectors_and_losses = sorted(
            vectors_and_losses, key=lambda x: x[1], reverse=True
        )[:len(untouched_centroids)]

        logger.info("Gathering vectors across all processes for re-initialization...")
        if dist.is_initialized():
            all_vectors_and_losses = [[] for _ in range(dist.get_world_size())]
            dist.all_gather_object(all_vectors_and_losses, vectors_and_losses)
            vectors_and_losses = [item for sublist in all_vectors_and_losses for item in sublist]

        # global sort
        vectors_and_losses = sorted(
            vectors_and_losses, key=lambda x: x[1], reverse=True
        )[:len(untouched_centroids)]

        # need to make sure all devices have the same vectors for consistency
        logger.info("Re-initializing centroids...")
        for centroid_idx, (idx, loss) in zip(untouched_centroids, vectors_and_losses):
            model.centroids.data[centroid_idx] = self.train_dataset.tensors[0][idx].clone().to(device=model.centroids.device)


class CentroidModel(nn.Module):
    def __init__(
            self,
            init_centroids: torch.Tensor,
            with_weighted_assignments: bool = False,
            training_Qs: torch.Tensor = None,  # ty:ignore[invalid-parameter-default]
            with_kmeans_objective: bool = False,
            max_difference_objective: bool = False,
            entropy_regularization: float = 0.0,
            use_soft_assignment: bool = False,
            initial_temperature: float = 1.0,
            final_temperature: float = 0.01,
            temperature_anneal_steps: int = 50000
        ):
        super(CentroidModel, self).__init__()
        self.centroids = nn.Parameter(init_centroids)
        self.with_weighted_assignments: bool = with_weighted_assignments
        
        self.training_Qs = training_Qs

        self.entropy_regularization: float = entropy_regularization
        if entropy_regularization > 0:
            self.register_buffer("target_dist", torch.ones(self.centroids.shape[0]) / self.centroids.shape[0])

        self.with_kmeans_objective: bool = with_kmeans_objective
        self.max_difference_objective: bool = max_difference_objective 

        # Temperature annealing parameters
        self.use_soft_assignment: bool = use_soft_assignment
        self.initial_temperature: float = initial_temperature
        self.final_temperature: float = final_temperature
        self.temperature_anneal_steps: int = temperature_anneal_steps
        self.register_buffer("current_step", torch.tensor(0))

    def get_temperature(self):
        if not self.use_soft_assignment or self.current_step >= self.temperature_anneal_steps:
            return self.final_temperature

        # cosine anneal from initial_temperature to final_temperature
        progress = self.current_step.float() / self.temperature_anneal_steps
        cos_factor = 0.5 * (1 + torch.cos(torch.pi * progress))
        temp = self.final_temperature + (self.initial_temperature - self.final_temperature) * cos_factor
        return temp

    def _calc_main_loss(self, queries: torch.Tensor, approx_dvecs: torch.Tensor, batch_dvecs: torch.Tensor):
        qc = queries @ approx_dvecs.T
        qd = queries @ batch_dvecs.T
        if not self.with_weighted_assignments:
            return torch.norm(qc - qd, dim=-1).mean()
        else:
            dr = batch_dvecs @ (batch_dvecs - approx_dvecs).T
            return torch.norm(qc*(1+dr) - qd, dim=-1).mean()

    def forward(self, batch_dvecs: torch.Tensor, is_training: bool = True, queries: torch.Tensor = None):
        similarities = batch_dvecs @ self.centroids.T  # [batch_size, n_centroids]

        if (not is_training) or (not self.training):
            assignments = similarities.argmax(dim=-1)
            if queries is None: # this is during encoding
                assert len(batch_dvecs.shape) == 3
                if self.with_weighted_assignments:
                    return assignments, (batch_dvecs * (batch_dvecs - self.centroids[assignments])).sum(-1)+1
                return (assignments, None)
            
            loss = self._calc_main_loss(queries, self.centroids[assignments], batch_dvecs)
            return loss, assignments

        if self.use_soft_assignment:
            temperature = self.get_temperature()
            soft_assignments = torch.softmax(similarities / temperature, dim=-1)
            centroids_dvecs = soft_assignments @ self.centroids

            if self.training:
                self.current_step += 1
        else:
            centroids_dvecs = self.centroids[similarities.argmax(dim=-1)]

        loss = self._calc_main_loss(
            queries=self.training_Qs if self.training_Qs is not None else batch_dvecs,
            approx_dvecs=centroids_dvecs,
            batch_dvecs=batch_dvecs
        )

        if self.max_difference_objective:
            # maximize the score gap between top-1 and top-2 centroid
            _, topk_centroid_idx = torch.topk(similarities, k=2, dim=-1)
            training_Qs = self.training_Qs if self.training_Qs is not None else batch_dvecs
            qc = training_Qs @ centroids_dvecs.T
            second_qc = training_Qs @ self.centroids[topk_centroid_idx[:, 1]].T
            loss = loss + (second_qc - qc).mean()

        if self.with_kmeans_objective:
            weighted_centroids = soft_assignments @ self.centroids
            kmeans_loss = 1 - torch.nn.functional.cosine_similarity(batch_dvecs, weighted_centroids, dim=-1).mean()
            loss = loss + kmeans_loss

        if self.entropy_regularization > 0:
            loss = loss + self.entropy_regularization * nn.functional.kl_div(
                torch.log(soft_assignments.mean(dim=0) + 1e-8),
                self.target_dist,
                reduction='batchmean'
            )

        return loss,

