from pathlib import Path
import json
import pickle
from typing import Any, Dict, Tuple, Callable

import numpy as np
import scipy.sparse as sp
from transformers.utils.logging import get_logger
from tqdm import tqdm

logger = get_logger(__name__)

def _make_full_sparse_matrix(raw: Dict[int, Dict[int, float]], n_centroids: int):
    ndocs = len(raw)
    assert ndocs-1 == max(raw.keys()), "Not all passages are loaded"
    assert 0 == min(raw.keys())

    indptr = np.array([0] + [ len(raw[i]) for i in range(ndocs) ], dtype=np.int64).cumsum()
    indices = np.array([ c for i in range(ndocs) for c in sorted(raw[i].keys()) ], dtype=np.int64)
    vals = np.array([
        v for i in range(ndocs) for _, v in sorted(raw[i].items(), key=lambda x: x[0])
    ], dtype=np.float16)

    assert indices.max() < n_centroids

    return sp.csr_matrix(
        (vals, indices, indptr),
        shape=(indptr.shape[0]-1, n_centroids),
        copy=False
    )


def _load_mmap_spm(path_prefix: str, nnz: int, shape: tuple, use_mmap: bool = True):
    indices = np.load(path_prefix + "_indices.npy", mmap_mode='r' if use_mmap else None)
    if Path(path_prefix + "_data.npy").exists():
        data = np.load(path_prefix + "_data.npy", mmap_mode='r' if use_mmap else None)
    else:
        data = indices
    indptr = np.load(path_prefix + "_indptr.npy")
    return sp.csr_matrix((data, indices, indptr), shape=shape, copy=False)

def _write_spm_as_mmap(spm: sp.csr_matrix, path_prefix: str):
    np.save(path_prefix + "_data.npy", spm.data)
    np.save(path_prefix + "_indices.npy", spm.indices)
    np.save(path_prefix + "_indptr.npy", spm.indptr)

def _is_main_process(local_rank: int):
    return local_rank < 1


class ColBERTSaRIndexer:
    def __init__(
            self,
            index_dir: str,
            n_centroids: int = None,
            write_freq: int = 10000,
            local_rank: int = 0,
            metadata: dict[str, Any] = None
        ):
        self.index_dir = Path(index_dir)
        self.n_centroids = n_centroids
        self.n_docs = -1
        self.nnz = -1
        self.total_shards = -1
        self.additional_info = metadata or {}

        self.buffer: dict[int, dict[int, float]] = {}
        self.local_rank = local_rank
        self.write_freq = write_freq

        self.shard_idx = 0

        if _is_main_process(local_rank):
            if (self.index_dir / "metadata.json").exists():
                self._load_metadata()
            else:
                self.index_dir.mkdir(parents=True, exist_ok=True)
                self._update_metadata()

    def _load_metadata(self):
        metadata = json.loads((self.index_dir / "metadata.json").read_text())
        assert self.n_centroids == metadata['n_centroids']
        if metadata['n_docs'] != -1:
            self.n_docs = metadata['n_docs']
        if metadata['nnz'] != -1:
            self.n_docs = metadata['nnz']
        if metadata['total_shards'] != -1:
            self.n_docs = metadata['total_shards']

    def _update_metadata(self):
        with open(self.index_dir / "metadata.json", "w") as fw:
            json.dump({
                "n_centroids": self.n_centroids,
                "n_docs": self.n_docs,
                "nnz": self.nnz,
                "total_shards": self.total_shards,
                **self.additional_info
            }, fw, indent=2)

    def update_total_n_shards(self):
        self.total_shards = len(list(self.index_dir.glob("centroid_shard_*.pkl")))
        self._update_metadata()

    def add_document(self, pid: int, sparse_vec: set[int] | dict[int, float]):
        if isinstance(sparse_vec, set):
            sparse_vec = {i: 1.0 for i in sparse_vec}

        assert all(0 <= c < self.n_centroids for c in sparse_vec.keys())
        assert pid not in self.buffer

        self.buffer[pid] = sparse_vec

        if len(self.buffer) >= self.write_freq:
            self.flush()

    def flush(self):
        if len(self.buffer) == 0:
            return

        with open(self.index_dir / f"centroid_shard_{self.shard_idx}.{self.local_rank}.pkl", "wb") as fw:
            pickle.dump(self.buffer, fw)

        self.buffer = {}
        self.shard_idx += 1

    def merge_shards(self, with_tqdm: bool = True, n_way: int = 8, n_workers: int = 8) -> Tuple[Dict[int, Dict[int, float]], Callable]:
        if not _is_main_process(self.local_rank):
            return {}, lambda x: x

        all_shard_files = sorted(self.index_dir.glob("centroid_shard_*.pkl"))

        all_buffer: dict[int, dict[int, float]] = {}

        for fn in tqdm(all_shard_files, dynamic_ncols=True, disable=not with_tqdm, desc="Merging shards"):
            with open(fn, "rb") as fr:
                shard: dict[int, dict[int, float]] = pickle.load(fr)
                all_buffer |= shard

        def cleanup():
            for shard_file in all_shard_files:
                shard_file.unlink()

        return all_buffer, cleanup

    def create_index(self, n_way: int = 8, n_workers: int = 8):
        if not _is_main_process(self.local_rank):
            return

        all_sparse_mapping, cleanup = self.merge_shards(n_way=n_way, n_workers=n_workers)
        forward_matrix = _make_full_sparse_matrix(all_sparse_mapping, n_centroids=self.n_centroids)
        logger.info("forward index created")
        _write_spm_as_mmap(forward_matrix, str(self.index_dir / "forward"))
        logger.info("forward index saved, creating inverted index")
        _write_spm_as_mmap(forward_matrix.T.tocsr(), str(self.index_dir / "inverted"))
        logger.info("inverted index created")

        self.n_docs = len(all_sparse_mapping)
        self.nnz = forward_matrix.nnz

        self._update_metadata()

        cleanup()
