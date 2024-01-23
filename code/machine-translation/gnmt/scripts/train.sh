#!/usr/bin/env bash

python -m gnmt.nmt \
    --src=de --tgt=en \
    --hparams_path=./gnmt/hyperparams/gnmt_4_layer.json \
    --out_dir=./out/gnmt_4_layer \
    --vocab_prefix=./gnmt/wmt16_de_en/vocab.bpe.32000  \
    --train_prefix=./gnmt/wmt16_de_en/train.tok.clean.bpe.32000 \
    --dev_prefix=./gnmt/wmt16_de_en/newstest2013.tok.bpe.32000  \
    --test_prefix=./gnmt/wmt16_de_en/newstest2015.tok.bpe.32000
