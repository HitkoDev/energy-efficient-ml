#!/usr/bin/env bash

models=("3_layer"  "gnmt_2_layer"  "gnmt_4_layer"  "gnmt_8_layer")

for model in ${models[*]}; do
    python -m gnmt.nmt \
        --src=en --tgt=de \
        --hparams_path=./gnmt/hyperparams/${model}.json \
        --out_dir=./out/${model} \
        --vocab_prefix=./gnmt/wmt16_de_en/vocab.bpe.32000  \
        --train_prefix=./gnmt/wmt16_de_en/train.tok.clean.bpe.32000 \
        --dev_prefix=./gnmt/wmt16_de_en/newstest2013.tok.bpe.32000  \
        --test_prefix=./gnmt/wmt16_de_en/newstest2015.tok.bpe.32000
done
