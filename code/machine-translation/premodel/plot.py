import matplotlib.pyplot as plt
import numpy as np

# to change default colormap
plt.rcParams["image.cmap"] = "Set3"
# to change default color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Set3.colors)

data = {
    "Nearest Neighbors": {
        "bleu": 20.871800948542578,
        "rouge": 47.88291338042365,
        "dur": 4.26466452821522,
        "f1": 29.071537760472143,
    },
    "Linear SVM": {
        "bleu": 20.78160167092803,
        "rouge": 47.79579823881847,
        "dur": 4.196983546222658,
        "f1": 28.96794692859158,
    },
    "RBF SVM": {
        "bleu": 20.791106572616172,
        "rouge": 47.80376482226238,
        "dur": 4.205156915036726,
        "f1": 28.978643702687762,
    },
    "Decision Tree": {
        "bleu": 21.220054752280838,
        "rouge": 48.24036795683671,
        "dur": 5.248131481013843,
        "f1": 29.47471983005598,
    },
    "Random Forest": {
        "bleu": 20.839257495261837,
        "rouge": 47.86659113944153,
        "dur": 4.376103914459113,
        "f1": 29.036952108074267,
    },
    "Neural Net": {
        "bleu": 20.86767689546544,
        "rouge": 47.87744930696139,
        "dur": 4.246399016139792,
        "f1": 29.066530179165667,
    },
    "AdaBoost": {
        "bleu": 20.904761012082975,
        "rouge": 47.91841297443884,
        "dur": 4.2657083981625155,
        "f1": 29.110048644519516,
    },
    "Naive Bayes": {
        "bleu": 20.83386475050523,
        "rouge": 47.86692906936817,
        "dur": 4.7656397286076935,
        "f1": 29.03177884284544,
    },
    "QDA": {
        "bleu": 20.84422349047537,
        "rouge": 47.860021236074274,
        "dur": 4.732432123529531,
        "f1": 29.040563152224596,
    },
}

overall = {
    "3_layer_bleu": 20.781824591127283,
    "3_layer_rouge": 47.79569826741286,
    "3_layer_f1": 28.968145135622212,
    "3_layer_dur": 4.196777612546452,
    "gnmt_2_layer_bleu": 21.79603912219318,
    "gnmt_2_layer_rouge": 48.77836108476305,
    "gnmt_2_layer_f1": 30.129198786025754,
    "gnmt_2_layer_dur": 6.5326479653892715,
    "gnmt_8_layer_bleu": 22.68979996925928,
    "gnmt_8_layer_rouge": 49.48964688946783,
    "gnmt_8_layer_f1": 31.11440270993239,
    "gnmt_8_layer_dur": 8.864679464644022,
    "oracle_dur": 7.145847593861133,
    "oracle_bleu": 26.384912912805746,
    "oracle_rouge": 52.834770163861535,
    "oracle_f1": 35.194304127484784,
}

items = [(k, v) for k, v in data.items()]
items.sort(key=lambda x: -x[1]["f1"])

best = items[0][1]
overall["our_bleu"] = best["bleu"]
overall["our_rouge"] = best["rouge"]
overall["our_dur"] = best["dur"]
overall["our_f1"] = best["f1"]

fig, ax = plt.subplots()
fig.set_figheight(3)
ax2 = ax.twinx()

species = [k for k, v in items]
x_pos = np.arange(len(species))
times = [v["dur"] for k, v in items]
f1s = [v["f1"] for k, v in items]

line1 = ax.bar(x_pos, times, align="center", label="inference")
ax.set_xticks(x_pos, labels=species)
ax.invert_xaxis()  # labels read top-to-bottom
ax.set_ylabel("Inference (ms)")
line2 = ax2.plot(x_pos, f1s, "-o", color="black", label="F1")
ax2.set_ylabel("F1")
ax.legend(handles=[line1, line2[0]], loc="upper left", ncols=2)

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

fig.tight_layout()

plt.show()


species = ("3_layer", "gnmt_2_layer", "gnmt_8_layer", "our", "oracle")
penguin_means = {
    "BLEU": [overall["%s_bleu" % v] for v in species],
    "ROUGE": [overall["%s_rouge" % v] for v in species],
    "F1": [overall["%s_f1" % v] for v in species],
}

x = np.arange(len(species))  # the label locations
width = 0.25  # the width of the bars
multiplier = 0

fig, ax = plt.subplots()
fig.set_figheight(3)

for attribute, measurement in penguin_means.items():
    offset = width * multiplier
    rects = ax.bar(x + offset, measurement, width, label=attribute)
    multiplier += 1

ax.set_xticks(x + width, species)
ax.legend(loc="upper left", ncols=3)

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

fig.tight_layout()

plt.show()

fig, ax = plt.subplots()
fig.set_figheight(3)

x_pos = np.arange(len(species))[::-1]
times = [overall["%s_dur" % v] for v in species]

ax.bar(x_pos, times, align="center")
ax.set_xticks(x_pos, labels=species)
ax.invert_xaxis()  # labels read top-to-bottom
ax.set_ylabel("Inference (ms)")

plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment="right")

fig.tight_layout()

plt.show()
