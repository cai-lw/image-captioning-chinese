from utils.load_data import *
from utils.text_process import *
from utils.metrics import *

val_idx, val_sentences = load_text('./data/valid.txt')
val_seq, vocab, vocab_inv = encode_text(val_sentences, vocab=None, ignore_non_chinese=False, with_begin_end=False)
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
rouge_l_sum = 0
cider_d_sum = 0
bleu_counter = BLEUCounter()
with open('agreement_test.txt', 'w', encoding='utf8') as f:
    for idx, seq in sorted(val_test.items(), key=lambda p:p[0]):
        bleu_counter.add(seq, val_dict[idx])
        rouge_l_sum += rouge_l(seq, val_dict[idx])
        cider_d_sum += cider_d(seq, val_dict[idx], idf, n_val)
        print(decode_text([seq], vocab_inv, sep=' ')[0], file=f)
bleus = bleu_counter.get_bleu()
for n in range(1, 5):
    print('[ VAL ]', 'BLEU-%d=%f'%(n, bleus[n - 1]))
print('[ VAL ]', 'METEOR=%f'%(meteor('agreement_test.txt', 'data/valid_meteor_agreement.txt', n_ref=4)))
print('[ VAL ]', 'ROUGE-L=%f'%(rouge_l_sum / n_val))
print('[ VAL ]', 'CIDEr-D=%f'%(cider_d_sum / n_val))
