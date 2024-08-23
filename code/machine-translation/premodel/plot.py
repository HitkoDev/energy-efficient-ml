import matplotlib.pyplot as plt
import numpy as np

# to change default colormap
plt.rcParams["image.cmap"] = "Set3"
# to change default color cycle
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=plt.cm.Set3.colors)

data = {
    "Nearest Neighbors": {
        "bleu": 21.406649968995364,
        "rouge": 48.390546021368294,
        "dur": 5.591267588637724,
        "f1": 29.682552881666133,
    },
    "Linear SVM": {
        "bleu": 21.072178283014896,
        "rouge": 48.074944664032984,
        "dur": 5.110819136162432,
        "f1": 29.30111222942296,
    },
    "RBF SVM": {
        "bleu": 20.79914851258859,
        "rouge": 47.82395592513132,
        "dur": 4.263659304561254,
        "f1": 28.99016504416679,
    },
    "Decision Tree": {
        "bleu": 21.28743390573019,
        "rouge": 48.316158864628434,
        "dur": 5.355403691380038,
        "f1": 29.55384908945522,
    },
    "Random Forest": {
        "bleu": 20.921389490970203,
        "rouge": 47.94549965972975,
        "dur": 4.55429880125245,
        "f1": 29.13116840591863,
    },
    "Neural Net": {
        "bleu": 21.371311926670103,
        "rouge": 48.38046122681615,
        "dur": 5.6891677375972005,
        "f1": 29.646670795286546,
    },
    "AdaBoost": {
        "bleu": 21.510813528973962,
        "rouge": 48.506402200671516,
        "dur": 5.852999535995352,
        "f1": 29.804446287294276,
    },
    "Naive Bayes": {
        "bleu": 20.834976624281275,
        "rouge": 47.868657249162354,
        "dur": 4.766409624447801,
        "f1": 29.033176226433575,
    },
    "QDA": {
        "bleu": 20.839856296432497,
        "rouge": 47.854068227786996,
        "dur": 4.732737795294599,
        "f1": 29.035228718520745,
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
