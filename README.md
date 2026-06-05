# ColBERTSaR: Sparsified ColBERT Index via Product Quantization

A product-quantization codebook is trained on top of a frozen ColBERT checkpoint;
every passage token is then assigned to its nearest codebook centroid, turning
the document into a sparse bag of centroid ids. Retrieval reduces to an
inverted-index lookup over the codebook, with an optional forward-index rerank
using the original ColBERT scores.

## Installation

Requires Python >= 3.10 and a CUDA-enabled PyTorch install for training and encoding
(search itself runs fine on CPU).

```bash
pip install -r requirements.txt
```

`datasets` is only needed if you load a document collection through the `hfds:`
prefix in `passaging.py`; uncomment the line in `requirements.txt` if so.

## Pipeline overview

The release exposes three stages, mirrored by three command-line entry points:

| Stage | Script | Output |
| --- | --- | --- |
| 1. Passaging | `passaging.py` | `collection_passages.tsv`, `mapping.tsv` |
| 2. Indexing  | `index.py`     | centroids, sparse shards, forward + inverted index |
| 3. Search    | `search.py`    | TREC run file, optionally IR-measures report |

`index.py` is resumable: each of its four sub-stages (sampling, centroid training,
encoding, merging) is skipped if its outputs already exist in `--output_dir`.
Logs at startup tell you what will run and what will be skipped.

## Quick start

The script `beir.sh` reproduces the full pipeline over the BEIR benchmark using
`answerdotai/answerai-colbert-small-v1` as the backbone:

```bash
bash beir.sh
```

Edit the configuration block at the top of `beir.sh` to change the model,
output paths, subset list, or `nprobe` sweep.

## Stage-by-stage usage

### 1. Passaging

Segment a document collection into fixed-length passages so that the ColBERT
encoder sees pieces no longer than its max sequence length.

```bash
python passaging.py \
  --doc_collections irds:beir/nfcorpus \
  --docid_field doc_id \
  --output_dir ./passages/nfcorpus \
  --passage_length 512 \
  --passage_stride 512 \
  --tokenizer answerdotai/answerai-colbert-small-v1 \
  --num_workers 32 \
  --overwrite
```

`--doc_collections` accepts any of:
- `irds:<dataset-id>` &nbsp;— an [ir_datasets](https://ir-datasets.com) dataset id
- `hfds:<repo>:<config>/<split>` &nbsp;— a HuggingFace `datasets` reference (requires the optional `datasets` install)
- one or more local JSONL paths (gzip-compatible)

The output directory contains two TSVs:
- `collection_passages.tsv` &nbsp;— `passage_idx \t passage_text` (integer pids assigned by the script)
- `mapping.tsv` &nbsp;— `passage_idx \t original_docid_chunkidx`

### 2. Indexing

Train per-corpus centroids on a sample of ColBERT vectors, then encode every
passage as the argmax centroid id of each of its token embeddings:

```bash
torchrun --nproc_per_node=8 index.py \
  --fp16 \
  --colbert_checkpoint answerdotai/answerai-colbert-small-v1 \
  --n_centroids 500000 \
  --max_steps 100000 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 2048 \
  --per_device_eval_batch_size 32 \
  --chunk_size 100000 \
  --collection ./passages/nfcorpus/collection_passages.tsv \
  --passage_mapping ./passages/nfcorpus/mapping.tsv \
  --output_dir ./index/nfcorpus \
  --clean_up_samples
```

Useful options:
- `--training_queries irds:<id> | <tsv> | in-batch` &nbsp;— source of query embeddings used in the centroid training objective. Defaults to `in-batch`, i.e. document embeddings act as queries.
- `--centroids <path>` &nbsp;— load pre-trained centroids and skip centroid training (e.g. reuse centroids from a larger corpus).
- `--init_centroids <path>` &nbsp;— warm-start centroid training from a checkpoint.
- `--with_weighted_assignments` &nbsp;— store a per-token weight alongside the centroid id; the search side will pick this up automatically via the index `metadata.json`.
- `--resume` &nbsp;— resume centroid training from a HuggingFace `Trainer` checkpoint inside `--output_dir`.

Outputs in `--output_dir`:
- `centroids.pt` &nbsp;— the trained centroid matrix `[n_centroids, dim]`
- `forward_{data,indices,indptr}.npy` &nbsp;— per-document sparse representation
- `inverted_{data,indices,indptr}.npy` &nbsp;— transpose, used at query time
- `metadata.json` &nbsp;— `n_centroids`, `n_docs`, `nnz`, and the paths of the source artifacts

### 3. Search

```bash
torchrun --nproc_per_node=8 search.py \
  --fp16 \
  --index_dir ./index/nfcorpus \
  --queries irds:beir/nfcorpus/test \
  --qrels   irds:beir/nfcorpus/test \
  --per_device_eval_batch_size 64 \
  --nprobe 8 \
  --use_forward_index \
  --search_output ./runs/nfcorpus_np8.trec
```

- `--nprobe` &nbsp;— number of nearest centroids per query token used for first-stage scoring.
- `--use_forward_index` &nbsp;— after the first-stage shortlist, rerank the top-`topk` documents using the full sparse representation. Recommended; gives a meaningful quality bump for small extra cost.
- `--load_full_index_to_memory` &nbsp;— bypass `mmap` and load the index fully into RAM (useful when the index lives on slow storage but the host has enough memory).
- `--qrels` accepts both `irds:<id>` and a local TREC-format file. Setting it triggers an evaluation pass using `ir_measures`; `--metrics` controls which measures are reported (defaults to `nDCG@10 nDCG@20 P@10 R@1000`).

Search can be run single-process, on a single node with `torchrun` (each rank handles a query slice, results are gathered to rank 0), or against an index living on shared storage.

## Data formats

**Local queries TSV**: `qid \t query_text`

**Local collection TSV** (used as `--collection`): `pid \t passage_text \t [optional title]`. `pid` must be an integer that matches the indexes in `mapping.tsv`.

**Mapping TSV** (used as `--passage_mapping`): `pid \t original_docid_chunkidx`. The final `_<int>` suffix is stripped at search time so that passage-level scores can be max-pooled back to the document level.

**Local qrels**: standard TREC `qid 0 docid relevance` format.

## Implementation notes

- Training uses HuggingFace `Trainer` so all of its standard flags work (`--save_steps`, `--save_total_limit`, `--resume_from_checkpoint`, gradient accumulation, mixed precision, multi-GPU via `torchrun`).
- The sparse index is stored as two CSR matrices (`forward`, `inverted`) memory-mapped via NumPy, which makes startup cheap regardless of index size.
- The pure-Python search path is intentionally short (~30 lines in `ColBERTSaRSearcher.search_code`); for large indexes a Cython/C++ kernel can be plugged in by reintroducing the `fast_search_qcode` extension we used internally.

## Citation

```bibtex
@inproceedings{sigir2026colbertsar,
	title={ColBERTSaR: Sparsified ColBERT Index via Product Quantization},
	author={Eugene Yang and Andrew Yates and Dawn Lawrie and James Mayfield and Saron Samuel and Rohan Jha},
	booktitle={Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR) (Short Paper) (Accepted)},
	year={2026},
	url={https://arxiv.org/abs/2606.05568}
}
```
