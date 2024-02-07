from glob import glob

import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

dev_ppl = []
dev_bleu = []
for f in glob("./out/gnmt_2_layer/train_log/*"):
    for e in summary_iterator(f):
        for v in e.summary.value:
            if v.tag == 'dev_ppl':
                dev_ppl.append((e.step, v.simple_value))
            if v.tag == 'dev_bleu':
                dev_bleu.append((e.step, v.simple_value))

paper_dev_ppl = []
paper_dev_bleu = []
for f in glob("./out_paper/gnmt_2_layer/train_log/*"):
    for e in summary_iterator(f):
        for v in e.summary.value:
            if v.tag == 'dev_ppl':
                paper_dev_ppl.append((e.step, v.simple_value))
            if v.tag == 'dev_bleu':
                paper_dev_bleu.append((e.step, v.simple_value))

dev_ppl = sorted(dev_bleu, key=lambda x: x[0])
dev_bleu = sorted(dev_ppl, key=lambda x: x[0])
paper_dev_ppl = sorted(paper_dev_ppl, key=lambda x: x[0])
paper_dev_bleu = sorted(paper_dev_bleu, key=lambda x: x[0])

fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot([x[0] for x in dev_ppl], [x[1] for x in dev_ppl], label='TF-NMT')
ax1.plot([x[0] for x in paper_dev_ppl], [x[1] for x in paper_dev_ppl], label='Paper')

ax2.plot([x[0] for x in dev_bleu], [x[1] for x in dev_bleu], label='TF-NMT')
ax2.plot([x[0] for x in paper_dev_bleu], [x[1] for x in paper_dev_bleu], label='Paper')

ax1.set_yscale('log')
ax1.set_xlabel('Step')
ax1.set_ylabel('Perplexity')

ax2.set_xlabel('Step')
ax2.set_ylabel('BLEU')

plt.legend()
fig.suptitle('Training metrics')
fig.set_figheight(3)
fig.tight_layout()
plt.savefig('tf-nmt_paper_training.pdf')
