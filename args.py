import sys
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, cast

from tqdm import tqdm
from transformers import HfArgumentParser, TrainingArguments, set_seed
from transformers.utils.logging import get_logger
from transformers.trainer_utils import is_main_process
import ir_datasets as irds

from utils import load_mapping, dataclass_to_dict

logger = get_logger(__name__)

@dataclass
class Arguments(TrainingArguments):
    colbert_checkpoint: Optional[str] = None
    centroids: Optional[str] = None
    index_dir: Optional[str] = None

    # collection
    collection: Optional[str] = None
    n_passages: Optional[int] = None
    id_field: str = "id"
    text_fields: List[str] = field(default_factory=lambda: ['text'])
    passage_mapping: Optional[str] = None

    # actions
    resume: Optional[bool] = False

    def __post_init__(self):
        if self.index_dir is not None:
            if self.output_dir is None:
                logger.info("Assuming `index_dir` as `output_dir` for transformers")
                self.output_dir = self.index_dir
            else:
                raise ValueError(f"Both `index_dir`({self.index_dir}) and `output_dir` ({self.output_dir}) are set. Only one should be presented.")
        else:
            self.index_dir = self.output_dir

        if self.save_total_limit is None:
            # no need to keep that many checkpoints
            self.save_total_limit = 1

        super().__post_init__()

    @staticmethod
    def read_queries(query_path: str):
        if Path(query_path).exists():
            return dict(line.strip().split("\t") for line in open(query_path))
        else:
            query_path = query_path.replace("irds:", "")
            return { q.query_id: q.default_text() for q in irds.load(query_path).queries }

    def get_collection_len(self):
        if self.n_passages is not None:
            return self.n_passages

        self.n_passages = sum(
            1 for _ in tqdm(
                open(self.passage_mapping or self.collection),
                desc='Counting lines',
                disable=not is_main_process(self.local_rank)
            )
        )
        return self.n_passages

    def get_pid_mapping(self, force_pid_to_int: bool = False, with_tqdm: bool = False):
        if self.passage_mapping is None:
            return None
        return load_mapping(self.passage_mapping, force_pid_to_int=force_pid_to_int, with_tqdm=with_tqdm)

    def get_collection_generator(self, need_mapper: bool = True, force_pid_to_int: bool = True):
        if self.passage_mapping is not None:
            logger.info("Using passage mapping")
            cast = int if force_pid_to_int else str
            pid_mapper = self.get_pid_mapping(force_pid_to_int).__getitem__ if need_mapper else None
            coll = ( (cast(l[0]), l[1]) for l in (x.strip().split("\t")[:2] for x in open(self.collection)))
        else:
            pid_mapper = lambda x: x  # noqa: E731
            coll = (
                (doc[self.id_field], " ".join(doc[field] for field in self.text_fields))
                for doc in map(json.loads, open(self.collection))  # noqa: F821
            )
        return coll, pid_mapper

def parse_args(cls: TrainingArguments, check_output_dir: bool = True, write_args: bool = True, print_args: bool = True):
    parser = HfArgumentParser(cls)

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        args = parser.parse_json_file(json_file=Path(sys.argv[1]).absolute())[0]
    else:
        args = parser.parse_args_into_dataclasses()[0]
    args = cast(TrainingArguments, args)

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    if is_main_process(args.local_rank):
        if check_output_dir:
            if output_dir.exists() and len(list(output_dir.iterdir())) > 1 and not args.overwrite_output_dir and not args.resume:
                raise ValueError(
                    f"Output directory ({args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir or --resume to overcome."
                )
            output_dir.mkdir(parents=True, exist_ok=True)

        if write_args:
            with open(output_dir / "args.json", "w") as fw:
                json.dump(dataclass_to_dict(args), fw, indent=4)

        if print_args:
            logger.info(f"Arguments: {json.dumps(dataclass_to_dict(args), indent=4)}")

    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        + f"distributed training: {bool(args.local_rank != -1)}"
    )

    return args
