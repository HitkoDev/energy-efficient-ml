#!/usr/bin/env bash

models=("3_layer"  "gnmt_2_layer"  "gnmt_4_layer"  "gnmt_8_layer")
inputs=("newstest2009.tok"  "newstest2010.tok"  "newstest2011.tok"  "newstest2012.tok"  "newstest2013.tok"  "newstest2014.tok"  "newstest2015.tok"  "newstest2016.tok")

for model in ${models[*]}; do
    for input in ${inputs[*]}; do
        mkdir -p ./translated_best/${model}
        TF_CPP_MIN_LOG_LEVEL=3 python -m gnmt.nmt \
            --src=en --tgt=de \
            --out_dir=./out/${model}/best_bleu \
            --infer_mode beam_search --beam_width 10 \
            --infer_batch_size 1024 \
            --vocab_prefix=./gnmt/wmt16_de_en/vocab.bpe.32000  \
            --inference_input_file=./gnmt/wmt16_de_en/${input}.bpe.32000.en \
            --inference_output_file=./translated_best/${model}/${input} \
            --inference_ref_file=./gnmt/wmt16_de_en/${input}.bpe.32000.de > ./translated_best/${model}/${input}.log
        mkdir -p ./translated/${model}
        TF_CPP_MIN_LOG_LEVEL=3 python -m gnmt.nmt \
            --src=en --tgt=de \
            --out_dir=./out/${model} \
            --infer_mode beam_search --beam_width 10 \
            --infer_batch_size 1024 \
            --vocab_prefix=./gnmt/wmt16_de_en/vocab.bpe.32000  \
            --inference_input_file=./gnmt/wmt16_de_en/${input}.bpe.32000.en \
            --inference_output_file=./translated/${model}/${input} \
            --inference_ref_file=./gnmt/wmt16_de_en/${input}.bpe.32000.de > ./translated/${model}/${input}.log
    done
done
