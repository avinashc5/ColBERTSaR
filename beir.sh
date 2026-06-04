#!/bin/bash
# Reproduce BEIR experiments end-to-end:
#   1. segment documents into passages
#   2. train per-corpus centroids and build the sparse index
#   3. search queries and write a TREC run
#
# Adjust the variables in the configuration block below to suit your setup.

set -e

# ---------- configuration ----------
COLBERT_CHECKPOINT="answerdotai/answerai-colbert-small-v1"

# Where to place the segmented passages collection
PASSAGES_ROOT="./collections/beir"
PASSAGE_LENGTH=512
PASSAGE_STRIDE=512

# Where to place experiment indexes and run files
EXP_PREFIX="./experiments/beir"

# Subsets to run
SUBSETS=(arguana climate-fever dbpedia-entity fever fiqa hotpotqa msmarco nfcorpus nq quora scidocs scifact trec-covid webis-touche2020)

# nprobe values to search with
NPROBES=(2 4 8 16)

# Hardware
NGPU=8
NCPU=32
# -----------------------------------


# 1) Passaging -- segment documents into fixed-length passages
for subset in "${SUBSETS[@]}"; do
    python passaging.py \
        --output_dir $PASSAGES_ROOT/$subset/passages/${PASSAGE_LENGTH}-${PASSAGE_STRIDE} \
        --passage_length $PASSAGE_LENGTH \
        --passage_stride $PASSAGE_STRIDE \
        --doc_collections irds:beir/$subset \
        --docid_field doc_id \
        --num_workers $NCPU \
        --overwrite
done


# 2) Train centroids and build the sparse index
for subset in "${SUBSETS[@]}"; do

    # large corpora use more centroids
    case "$subset" in
        trec-covid|climate-fever|dbpedia-entity|fever|hotpotqa|msmarco|nq)
            nc=1000000;;
        *)
            nc=500000;;
    esac

    output_dir="${EXP_PREFIX}/beir-${subset}_in-batch_cen${nc}_step100000"

    torchrun --nproc_per_node=$NGPU index.py \
        --fp16 \
        --n_centroids $nc \
        --colbert_checkpoint $COLBERT_CHECKPOINT \
        --max_steps 100000 \
        --learning_rate 1e-4 \
        --per_device_train_batch_size 2048 \
        --per_device_eval_batch_size 32 \
        --save_total_limit 2 \
        --save_steps 1000 \
        --chunk_size 100000 \
        --output_dir $output_dir \
        --collection $PASSAGES_ROOT/$subset/passages/${PASSAGE_LENGTH}-${PASSAGE_STRIDE}/collection_passages.tsv \
        --passage_mapping $PASSAGES_ROOT/$subset/passages/${PASSAGE_LENGTH}-${PASSAGE_STRIDE}/mapping.tsv \
        --clean_up_samples \
        --resume

done


# 3) Search
for subset in "${SUBSETS[@]}"; do

    case "$subset" in
        trec-covid|climate-fever|dbpedia-entity|fever|hotpotqa|msmarco|nq)
            nc=1000000;;
        *)
            nc=500000;;
    esac

    # not every BEIR subset has a /test split in ir_datasets
    case "$subset" in
        nfcorpus|scifact|fiqa|quora|dbpedia-entity|fever|hotpotqa|msmarco)
            qsubset="$subset/test";;
        *)
            qsubset="$subset";;
    esac

    index_dir="${EXP_PREFIX}/beir-${subset}_in-batch_cen${nc}_step100000"

    for nprobe in "${NPROBES[@]}"; do
        torchrun --nproc_per_node=$NCPU search.py \
            --fp16 \
            --index_dir $index_dir \
            --queries irds:beir/$qsubset \
            --qrels irds:beir/$qsubset \
            --per_device_eval_batch_size 64 \
            --nprobe $nprobe \
            --use_forward_index \
            --search_output ${index_dir}_np${nprobe}.trec
    done

done
