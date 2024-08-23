import os
import re
from glob import glob

import pandas as pd
from rouge import Rouge
from sacrebleu.metrics import BLEU

bleu_scorer = BLEU(effective_order=True)
rouge_scorer = Rouge()

base_dir = f"{os.path.dirname(__file__)}/../gnmt/wmt16_de_en"

pm15 = []
pm16 = []
premodels = {}
premodels_l = 0

for file in glob(f"{base_dir}/*.tok.en"):
    basename = os.path.basename(file)[:-3]

    # Loading gt sentences
    labels_file_path = f"{file[:-3]}.de"

    # core translated sentences against gt
    sents = []
    models = []
    models_ps = []
    for tr in glob(f"{os.path.dirname(__file__)}/../translated_best/*/{basename}"):
        log = tr+'.log'
        dur = 1
        with open(log, 'r', encoding="utf-8") as lg:
            for line in lg:
                m = re.search('num sentences (\d+), .*? time ([\d\.]+)ms', line)
                if m:
                    dur = float(m.group(2)) / float(m.group(1))
        k = os.path.basename(os.path.dirname(tr))
        if "4_layer" in k:
            continue
        with open(
            labels_file_path, "r", newline="\n", encoding="utf-8"
        ) as file_labels, open(tr, "r", encoding="utf-8") as file_tr:
            tok_sentences = [line.strip() for line in file_labels]
            tr_sentences = [re.sub("@@ ", "", line.strip()) for line in file_tr]

        for i, (l, t) in enumerate(zip(tok_sentences, tr_sentences)):
            try:
                bl_s = bleu_scorer.sentence_score(
                    hypothesis=t,
                    references=[l],
                ).score
                ro_s = (
                    rouge_scorer.get_scores(
                        hyps=t,
                        refs=l,
                    )[0][
                        "rouge-l"
                    ]["f"]
                    * 100
                )
            except:
                bl_s = 0
                ro_s = 0
            f1_s = 2 * ro_s * bl_s
            if ro_s + bl_s > 0:
                f1_s = f1_s / (ro_s + bl_s)
            if i >= len(sents):
                sents.append({})
                models.append([])
                models_ps.append([])
            sents[i][f"{k}_bleu"] = bl_s
            sents[i][f"{k}_rouge"] = ro_s
            sents[i][f"{k}_f1"] = f1_s
            sents[i][f"{k}_dur"] = dur
            models[i].append((bl_s, ro_s, f1_s, k, dur))
            models_ps[i].append((bl_s**2 / dur, ro_s**2 / dur, f1_s**2 / dur, k, dur))

    # Determine best model and oracle
    for i, m in enumerate(models):
        m.sort(key=lambda x: x[0])
        oracle = m[-1][3]
        sents[i]["oracle_dur"] = m[-1][4]
        models_ps[i].sort(key=lambda x: x[0])
        sents[i]["best_model"] = models_ps[i][-1][3]
        for k in list(sents[i].keys()):
            if oracle in k:
                sents[i][k.replace(oracle, "oracle")] = sents[i][k]

    if len(sents) > 0:
        score = {}
        for k in sents[0]:
            if "bleu" in k or "f1" in k or "rouge" in k or "dur" in k:
                score[k] = 0
                if k not in premodels:
                    premodels[k] = 0

        for v in sents:
            for k in v:
                if "bleu" in k or "f1" in k or "rouge" in k or "dur" in k:
                    score[k] += v[k]
                    premodels[k] += v[k]

        premodels_l += len(sents)

        for k in score:
            score[k] /= len(sents)
        for k in score:
            if "f1" in k:
                b = k.replace("f1", "bleu")
                r = k.replace("f1", "rouge")
                score[k] = 2 * score[b] * score[r] / (score[b] + score[r])
        print(basename, score)

        # Use 2015 and 2016 data for the premodels
        if "2015" in basename:
            pm15 = sents
        if "2016" in basename:
            pm16 = sents

        df = pd.DataFrame(sents)
        df.to_csv(
            f"{os.path.dirname(__file__)}/../translated_best/{basename}.csv", index=False
        )

for k in premodels:
    premodels[k] /= premodels_l
for k in premodels:
    if "f1" in k:
        b = k.replace("f1", "bleu")
        r = k.replace("f1", "rouge")
        premodels[k] = 2 * premodels[b] * premodels[r] / (premodels[b] + premodels[r])
print('overall', premodels)

df = pd.DataFrame(pm15 + pm16)
df.to_csv(f"{os.path.dirname(__file__)}/../translated_best/premodels.tok.csv", index=False)
