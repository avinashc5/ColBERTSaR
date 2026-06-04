from pathlib import Path
from tqdm import tqdm
import re

import gzip
import json

import logging
from multiprocessing import Pool
from functools import partial

import utils  # noqa: F401  -- triggers project-wide logging config

def strip_newlines(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text

def _read_doc_jsonl(corpus: str):
    fns = [ f for p in map(Path, corpus) for f in p.parent.glob(p.name) ]
    for fn in tqdm(fns, desc="processing file..."):
        opener = (gzip.open if fn.name.endswith('.gz') else open)
        num_docs = sum(1 for _ in tqdm(opener(fn), desc='counting'))
        with opener(fn) as f:
            for i, line in tqdm(enumerate(f), total=num_docs):
                try:
                    yield json.loads(line)
                except json.decoder.JSONDecodeError:
                    print(f"json decode error on line #{i} -- `{line}`")
                    continue


def create_doc_reader(corpus: str):
    if corpus[0].startswith('irds:'):
        import ir_datasets as irds
        assert len(corpus) == 1
        ds = irds.load(corpus[0].replace("irds:", ""))
        return (x._asdict() for x in ds.docs), len(ds.docs)

    if corpus[0].startswith('hfds:'):
        assert len(corpus) == 1
        from datasets import load_dataset
        dsstring: str = corpus[0].replace("hfds:", "")
        if dsstring.count('/') == 1:
            ds = load_dataset(*dsstring.split(":"))
            assert len(ds) == 1, f"No subset specified, there are `{ds.keys()}`"
            logging.warning(f"Use default `{list(ds.keys())[0]}` subset")
            ds = list(ds.values())[0]
        elif dsstring.count('/') == 2:
            dsstring, subset = dsstring.rsplit("/", 1)
            ds = load_dataset(*dsstring.split(":"))[subset]
        else:
            raise ValueError(f"Supported string `{corpus[0]}`")
        return ds, len(ds)

    return _read_doc_jsonl(corpus), None


def _process_document_batch(doc_batch, args_dict):
    from transformers import AutoTokenizer

    # tokenizers are not picklable, so each worker rebuilds its own
    tokenizer = AutoTokenizer.from_pretrained(args_dict['tokenizer_name'], use_fast=True)

    results = []
    for doc in doc_batch:
        doc_id = doc[args_dict['docid']]
        title = strip_newlines(doc[args_dict['title']]) if args_dict['title'] in doc else ""
        text = strip_newlines("  ".join(doc[field] for field in args_dict['bodys']))
        doc_text = title + " " + text if title else text
        if args_dict['lower']:
            doc_text = doc_text.lower()

        encoded_tokens = tokenizer.encode(doc_text)[1:-1]
        s, e, idx = 0, 0, 0
        while s < len(encoded_tokens):
            e = s + args_dict['length']
            if e >= len(encoded_tokens):
                e = len(encoded_tokens)
            p = tokenizer.decode(encoded_tokens[s:e])
            pass_id = f"{doc_id}_{idx}"
            s = s + args_dict['stride']
            results.append((p, pass_id))
            idx += 1

    return results


def create_passage_collection(
        output_dir: str,
        doc_collections: list[str],
        tokenizer: str,
        docid_field: str = "id",
        title_field: str = "title",
        body_fields: list[str] = ["text"],
        lowercase: bool = False,
        passage_length: int = 180,
        passage_stride: int = 90,
        num_workers: int = 8,
        doc_batch_size: int = 100,
        max_buffered_passages: int = 10000,
        overwrite: bool = False
    ):
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

    logging.info(
        f"Passaging {doc_collections} -> {output_dir} "
        f"(length={passage_length}, stride={passage_stride}, tokenizer={tokenizer}, workers={num_workers})"
    )

    output_dir: Path = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=overwrite)

    args_dict = {
        'tokenizer_name': tokenizer,
        'docid': docid_field,
        'title': title_field,
        'bodys': body_fields,
        'lower': lowercase,
        'length': passage_length,
        'stride': passage_stride,
    }

    pass_file = output_dir / "collection_passages.tsv"
    map_file = output_dir / "mapping.tsv"

    reader, total_docs = create_doc_reader(doc_collections)

    passage_idx = 0
    passage_buffer = []

    with open(pass_file, "w") as f, open(map_file, "w") as g:
        with Pool(num_workers) as pool:
            worker_fn = partial(_process_document_batch, args_dict=args_dict)
            doc_iterator = tqdm(reader, desc="Reading documents", total=total_docs)

            def batch_generator():
                batch = []
                for doc in doc_iterator:
                    batch.append(doc)
                    if len(batch) >= doc_batch_size:
                        yield batch
                        batch = []
                if batch:
                    yield batch

            for passages_batch in pool.imap_unordered(worker_fn, batch_generator(), chunksize=1):
                passage_buffer.extend(passages_batch)

                if len(passage_buffer) >= max_buffered_passages:
                    for passage_text, passage_id in passage_buffer:
                        f.write(f"{passage_idx}\t{passage_text}\n")
                        g.write(f"{passage_idx}\t{passage_id}\n")
                        passage_idx += 1
                    passage_buffer = []

            for passage_text, passage_id in passage_buffer:
                f.write(f"{passage_idx}\t{passage_text}\n")
                g.write(f"{passage_idx}\t{passage_id}\n")
                passage_idx += 1

    logging.info(f"Wrote {passage_idx} passages to {pass_file} and mapping to {map_file}")

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--doc_collections", type=str, nargs='+', required=True)
    parser.add_argument("--passage_length", type=int, default=180)
    parser.add_argument("--passage_stride", type=int, default=90)
    parser.add_argument("--docid_field", type=str, default="id")
    parser.add_argument("--title_field", type=str, default="title")
    parser.add_argument("--body_fields", type=str, nargs="+", default=["text"])
    parser.add_argument("--lowercase", action="store_true", default=False)
    parser.add_argument("--tokenizer", type=str, default="xlm-roberta-large")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--doc_batch_size", type=int, default=100)
    parser.add_argument("--max_buffered_passages", type=int, default=10000)
    parser.add_argument("--overwrite", action="store_true", default=False)

    create_passage_collection(**vars(parser.parse_args()))
