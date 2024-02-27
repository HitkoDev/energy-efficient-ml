import os
import re
from glob import glob

import pandas as pd
from bleu import compute_bleu
from rouge import rouge


def _bleu(ref_file, trans_file):
    max_order = 4
    smooth = False

    reference_text = []
    reference_text.append(ref_file)

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    translations = []
    for line in trans_file:
        translations.append(line.split(" "))

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = compute_bleu(
        per_segment_references, translations, max_order, smooth
    )
    return 100 * bleu_score


def _rouge(ref_file, trans_file):
    references = ref_file

    hypotheses = trans_file

    rouge_score_map = rouge(hypotheses, references)
    return 100 * rouge_score_map["rouge_l/f_score"]


base_dir = f"{os.path.dirname(__file__)}/../gnmt/wmt16_de_en"

pm15 = []
pm16 = []

for file in glob(f"{base_dir}/*.tok.en"):
    basename = os.path.basename(file)[:-3]

    # Loading gt sentences
    labels_file_path = f"{file[:-3]}.de"

    # core translated sentences against gt
    sents = []
    models = []
    models_ps = []
    for tr in glob(f"{os.path.dirname(__file__)}/../translated/*/{basename}"):
        duration = pd.read_csv(f"{os.path.dirname(tr)}/duration.csv")
        k = os.path.basename(os.path.dirname(tr))
        with open(
            labels_file_path, "r", newline="\n", encoding="utf-8"
        ) as file_labels, open(tr, "r", encoding="utf-8") as file_tr:
            tok_sentences = [line.strip() for line in file_labels]
            tr_sentences = [re.sub("@@ ", "", line.strip()) for line in file_tr]
        dur = duration[basename][0] / len(tok_sentences) / 1000

        for i, (l, t) in enumerate(zip(tok_sentences, tr_sentences)):
            r = rouge([t], [l])

            bl_s = _bleu([l], [t])
            ro_s = _rouge([l], [t])
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
            models[i].append((bl_s, ro_s, f1_s, k, dur))
            models_ps[i].append((bl_s**2 / dur, ro_s**2 / dur, f1_s**2 / dur, k))

    # Determine best model and oracle
    for i, m in enumerate(models):
        m.sort(key=lambda x: (x[0], x[1], x[2]))
        oracle = m[-1][3]
        sents[i]["oracle_dur"] = m[-1][4]
        models_ps[i].sort(key=lambda x: (x[0], x[1], x[2]))
        sents[i]["best_model"] = models_ps[i][-1][3]
        for k in list(sents[i].keys()):
            if oracle in k:
                sents[i][k.replace(oracle, "oracle")] = sents[i][k]

    if len(sents) > 0:
        score = {}
        for k in sents[0]:
            if "bleu" in k or "f1" in k or "rouge" in k:
                score[k] = 0

        for v in sents:
            for k in v:
                if "bleu" in k or "f1" in k or "rouge" in k:
                    score[k] += v[k]

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
            f"{os.path.dirname(__file__)}/../translated/{basename}.csv", index=False
        )

df = pd.DataFrame(pm15 + pm16)
df.to_csv(f"{os.path.dirname(__file__)}/../translated/premodels.tok.csv", index=False)
