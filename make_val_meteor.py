from utils.load_data import load_text
from utils.text_process import segmentation
import numpy as np

val_idx, val_sentences = load_text('./data/valid.txt')
val_seg = [' '.join(segmentation(s, ignore_non_chinese=False)) for s in val_sentences]

val_test = {}
val_dict = {}
for idx, seq in zip(val_idx, val_seg):
    if val_dict.get(idx) is None:
        val_dict[idx] = [seq]
        #val_test[idx] = seq
    else:
        val_dict[idx].append(seq)
for idx in val_dict.keys():
    if len(val_dict[idx]) == 0:
        val_dict[idx] = [val_test[idx]]
    if len(val_dict[idx]) < 5:
        orig = val_dict[idx][:]
        shuffle_idx = np.random.permutation(len(orig))
        for i in range(5 - len(orig)):
            val_dict[idx].append(orig[shuffle_idx[i % len(orig)]])
with open('./data/valid_meteor.txt', 'w', encoding='utf8') as f:
    for idx in sorted(val_dict.keys()):
        for s in val_dict[idx]:
            print(s, file=f)

