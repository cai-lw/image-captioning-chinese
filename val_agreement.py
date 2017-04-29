from utils.load_data import *
from utils.text_process import *
from utils.metrics import *

train_idx, train_sentences = load_text_seg('./data/train.txt')
val_idx, val_sentences = load_text_seg('./data/valid.txt')
train_seq, vocab, vocab_inv = encode_text(train_sentences, max_size=5000)
val_seq = encode_text(val_sentences, vocab)
val_test = {}
val_dict = {}
for idx, seq in zip(val_idx, val_seq):
    if val_dict.get(idx) is None:
        val_test[idx] = seq
        val_dict[idx] = []
    else:
        val_dict[idx].append(seq)
for idx in val_dict.keys():
    if len(val_dict[idx]) == 0:
        val_dict[idx] = [val_test[idx]]
idf = build_idf(val_dict)

n_val = 1000
bleu_sum = [0] * 4
rouge_l_sum = 0
cider_d_sum = 0
for idx, seq in val_test.items():
    for n in range(1, 5):
        bleu_sum[n - 1] += bleu(seq, val_dict[idx], n)
    rouge_l_sum += rouge_l(seq, val_dict[idx])
    cider_d_sum += cider_d(seq, val_dict[idx], idf, n_val)
for n in range(1, 5):
    print('[ VAL ]', 'BLEU-%d=%f'%(n, bleu_sum[n - 1] / n_val))
print('[ VAL ]', 'ROUGE-L=%f'%(rouge_l_sum / n_val))
print('[ VAL ]', 'CIDEr-D=%f'%(cider_d_sum / n_val))