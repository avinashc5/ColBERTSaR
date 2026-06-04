import os
from pathlib import Path
import logging
from typing import Dict
from dataclasses import is_dataclass, asdict

from tqdm.auto import tqdm

def write_trec_run(doc_scores: Dict[str, Dict[str, float]], fn: str, run_id: str = 'PLAID', topk=1000):
    Path(fn).parent.mkdir(parents=True, exist_ok=True)
    with open(fn, 'w') as fw:
        for qid, d in doc_scores.items():
            for i, (did, s) in enumerate(sorted(d.items(), key=lambda x:x[1], reverse=True)[:1000]):
                fw.write(f"{qid} 0 {did} {i+1} {s} {run_id}\n")

def read_trec_run(fn) -> Dict[str, Dict[str, float]]:
    scores = {}
    with open(fn) as fr:
        for line in fr:
            qid, _, did, _, s, _ = line.strip().split()
            if qid not in scores:
                scores[qid] = {}
            scores[qid][did] = float(s)
    return scores

def load_mapping(f: str, with_dpidx: bool=False, force_pid_to_int: bool = False, with_tqdm: bool = False) -> Dict[int, str]:  # noqa: F821
    cast = int if force_pid_to_int else str
    return {
        cast(pid): "_".join(tag.split("_")[:-1]) if not with_dpidx else tag
        for pid, tag in (l.strip().split("\t") for l in tqdm(open(f), dynamic_ncols=True, disable=not with_tqdm))
    }

def maxp(passage_scores: Dict[int, float], mapping: Dict[str, str] = None, limit: int = 1000) -> Dict[str, float]:
    doc_score = {}
    for pid, score in passage_scores.items():
        doc_id = mapping[pid] if mapping else pid
        if doc_id not in doc_score or doc_score[doc_id] < score:
            doc_score[doc_id] = score
    if len(doc_score) > limit:
        doc_score = dict(sorted(doc_score.items(), key=lambda x:x[1], reverse=True)[:limit])
    return doc_score

def batching(it, bs: int):
    while True:
        ret = [ e for _, e in zip(range(bs), it) ]
        if len(ret) == 0:
            return
        yield ret

def split_by_rank(it, local_rank, world_size):
    for i, e in enumerate(it):
        if i % world_size == local_rank:
            yield e

def dataclass_to_dict(obj):
    assert is_dataclass(obj)
    return asdict(obj)


logging.basicConfig(
    format="%(asctime)s [%(levelname)s][%(name)s:%(lineno)d] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO if int(os.environ.get('LOCAL_RANK', 0)) < 1 else logging.WARN,
)
