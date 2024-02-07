#!/usr/bin/env bash

models=("gnmt_2_layer")

for model in ${models[*]}; do
    python -m gnmt.nmt \
        --src=en --tgt=de \
        --hparams_path=./gnmt/hyperparams/${model}.json \
        --out_dir=./out_paper/${model} \
        --vocab_prefix=./gnmt/wmt16_de_en_paper/vocab.bpe.32000  \
        --train_prefix=./gnmt/wmt16_de_en_paper/train.tok.clean.bpe.32000 \
        --dev_prefix=./gnmt/wmt16_de_en_paper/newstest2015.tok.bpe.32000  \
        --test_prefix=./gnmt/wmt16_de_en_paper/newstest2016.tok.bpe.32000
done
