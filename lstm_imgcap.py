import tensorflow as tf
import tensorlayer as tl
import numpy as np

from utils.load_data import *
from utils.text_process import *
from utils.metrics import *

d_feature = 4096

n_hidden = 512
n_embed = 512
keep_prob = 0.5
k_beam = 3
use_full_val = True

train_img, val_img, test_img = load_img('./data/image_vgg19_fc2_feature.h5')
n_val = val_img.shape[0]
n_test = test_img.shape[0]
train_idx, train_sentences = load_text_seg('./data/train.txt')
train_dict = {}
for idx, stc in zip(train_idx, train_sentences):
    train_dict.setdefault(idx, []).append(stc)
for idx in train_dict.keys():
    if len(train_dict[idx]) < 5:
        orig = train_dict[idx][:]
        for i in range(5 - len(orig)):
            train_dict[idx].append(orig[np.random.randint(len(orig))])
train_pairs = [(idx, stc) for idx, stcs in train_dict.items() for stc in stcs]
train_idx = [idx for idx, stc in train_pairs]
train_sentences = [stc for idx, stc in train_pairs]
train_seq, vocab, vocab_inv = encode_text(train_sentences)

val_idx, val_sentences = load_text_seg('./data/valid.txt')
val_seq = encode_text(val_sentences, vocab)
# val_seq = val_sentences
val_dict = {}
val_test = {}
for idx, seq in zip(val_idx, val_seq):
    if val_dict.get(idx) is None:
        if use_full_val:
            val_dict[idx] = [seq]
        else:
            val_test[idx] = seq
            val_dict[idx] = []
    else:
        val_dict[idx].append(seq)
for idx in val_dict.keys():
    if len(val_dict[idx]) == 0:
        val_dict[idx] = [val_test[idx]]

idf = build_idf(val_dict)


train_seq, train_len = seq2array(train_seq)
max_step = train_seq.shape[1]
n_train = len(train_idx)
train_more_img = np.zeros((n_train, d_feature), dtype=np.float32)
for i in range(len(train_idx)):
    train_more_img[i, :] = train_img[train_idx[i] - 1, :]

sess = tf.InteractiveSession()

img = tf.placeholder(tf.float32, shape=[None, d_feature], name='img')
seq_in = tf.placeholder(tf.int64, shape=[None, max_step - 1], name='seq_in')
seq_len = tf.placeholder(tf.int64, shape=[None], name='seq_len')
seq_mask = tf.placeholder(tf.float32, shape=[None, max_step - 1], name='seq_mask')
seq_truth = tf.placeholder(tf.int64, shape=[None, max_step - 1], name='seq_truth')

init_net = tl.layers.InputLayer(inputs=img)
init_net = tl.layers.DenseLayer(init_net,
    n_units=n_hidden,
    act=tf.sigmoid,
    name='init_transform')

network = tl.layers.EmbeddingInputlayer(
    inputs=seq_in,
    vocabulary_size=len(vocab),
    embedding_size=n_embed,
    name='embedding')
network = tl.layers.DropoutLayer(network, keep=keep_prob, name='lstm_in_dropout')
init_state_c = tf.placeholder(tf.float32, shape=[None, n_hidden], name='init_state_c')
init_state_h = tf.placeholder_with_default(init_net.outputs, shape=[None, n_hidden], name='init_state_h')
network = tl.layers.DynamicRNNLayer(network,
    cell_fn=tf.contrib.rnn.BasicLSTMCell,
    n_hidden=n_hidden,
    sequence_length=seq_len,
    initial_state=tf.contrib.rnn.LSTMStateTuple(init_state_c, init_state_h),
    return_seq_2d=True,
    name='lstm')
state_outputs = network.final_state
network = tl.layers.DropoutLayer(network, keep=keep_prob, name='lstm_out_dropout')
network = tl.layers.DenseLayer(network,
    n_units=len(vocab),
    act=tf.identity,
    name='unembedding')
network = tl.layers.ReshapeLayer(network, shape=[-1, max_step - 1, len(vocab)])

k_top = tf.placeholder_with_default(1, shape=[], name='k_top')
top_k_loglike, top_k_ind = tf.nn.top_k(tf.nn.log_softmax(network.outputs), k=k_top)
loss_per_word = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seq_truth, logits=network.outputs, name='cross_entropy')
loss = tf.reduce_sum(loss_per_word * seq_mask)

dropout_dict = {**network.all_drop, **init_net.all_drop}
train_vars = network.all_params + init_net.all_params
train_op = tf.train.AdamOptimizer().minimize(loss, var_list=train_vars)


def generate_seq(batch_img, k_beam=1):
    batch_size = batch_img.shape[0]
    first_dim = batch_size * k_beam
    batch_img_in = np.repeat(batch_img, k_beam, axis=0)
    batch_seq_in = np.zeros((first_dim, max_step - 1), dtype=np.int64)
    batch_seq_len = np.ones((first_dim,), dtype=np.int64)
    batch_res = np.zeros((first_dim, max_step - 1), dtype=np.int64)
    batch_loglike_sum = np.zeros((first_dim,), dtype=np.float32)
    zero_state = np.zeros((first_dim, n_hidden), dtype=np.float32)
    for t in range(max_step - 1):
        if t == 0:
            batch_seq_in[:, 0] = vocab['<BEG>']
            feed_dict = {img: batch_img_in, k_top: k_beam,
                         seq_in: batch_seq_in, seq_len: batch_seq_len, init_state_c: zero_state}
        else:
            batch_seq_in[:, 0] = new_step
            feed_dict = {img: batch_img_in, k_top: k_beam,
                         seq_in: batch_seq_in, seq_len: batch_seq_len, init_state_c: new_c, init_state_h: new_h}
        feed_dict.update(tl.utils.dict_to_one(dropout_dict))
        step_loglike, step_ind, step_state = sess.run([top_k_loglike, top_k_ind, state_outputs], feed_dict=feed_dict)

        new_batch_res = np.zeros((first_dim, max_step - 1), dtype=np.int64)
        new_batch_loglike_sum = np.zeros((first_dim,), dtype=np.float32)
        new_step = np.zeros((first_dim,), dtype=np.int64)
        new_c = np.zeros((first_dim, n_hidden), dtype=np.float32)
        new_h = np.zeros((first_dim, n_hidden), dtype=np.float32)
        for i in range(batch_size):
            if t > 0:
                for j in range(k_beam):
                    if batch_res[i * k_beam + j, t - 1] == vocab['<END>']:
                        step_loglike[i * k_beam + j, 0, 0] = -np.log(4)
                        step_loglike[i * k_beam + j, 0, 1:] = -np.inf
                        step_ind[i * k_beam + j, 0, 0] = vocab['<END>']
            cand_loglike = batch_loglike_sum[i * k_beam:(i + 1) * k_beam].reshape((-1, 1)) \
                           + step_loglike[i * k_beam:(i + 1) * k_beam, 0, :]
            if t == 0:
                cand_top_k = np.arange(k_beam)
            else:
                cand_top_k = np.argsort(cand_loglike, axis=None)[-1:-k_beam-1:-1]
            for rank, ind in enumerate(cand_top_k):
                row = ind // k_beam
                col = ind % k_beam
                new_batch_res[i * k_beam + rank, :t] = batch_res[i * k_beam + row, :t]
                new_step[i * k_beam + rank] = step_ind[i * k_beam + row, 0, col]
                new_batch_loglike_sum[i * k_beam + rank] = cand_loglike[row, col]
                new_c[i * k_beam + rank] = step_state.c[i * k_beam + row, :]
                new_h[i * k_beam + rank] = step_state.h[i * k_beam + row, :]
        new_batch_res[:, t] = new_step
        batch_res = new_batch_res
        batch_loglike_sum = new_batch_loglike_sum
        all_end = np.all(np.any(batch_res == vocab['<END>'], axis=1))
        if all_end:
            # print(decode_text(array2seq(batch_res[2*k_beam:3*k_beam, :], vocab['<END>']), vocab_inv))
            # print(batch_loglike_sum[2*k_beam:3*k_beam])
            break
    best_res = batch_res[::k_beam, :]
    seqs = array2seq(best_res, vocab['<END>'])
    texts = decode_text(seqs, vocab_inv)
    return seqs, texts

sess.run(tf.global_variables_initializer())

batch_size = 20
zero_state = np.zeros((batch_size, n_hidden), dtype=np.float32)
max_epoch = 10
batch_per_epoch = n_train // batch_size
# batch_per_epoch = 100
n_batch_val = n_val // batch_size
n_batch_test = n_test // batch_size
accum_loss = 0
shuffle_idx = np.random.permutation(n_train)
for i_epoch in range(max_epoch):
    for i_batch in range(batch_per_epoch):
        batch_idx = shuffle_idx[i_batch * batch_size:(i_batch + 1) * batch_size]
        batch_img = train_more_img[batch_idx, :]
        batch_seq_in = train_seq[batch_idx, :-1]
        batch_seq_truth = train_seq[batch_idx, 1:]
        batch_seq_len = train_len[batch_idx] - 1
        batch_seq_mask = np.zeros((batch_size, max_step - 1), dtype=np.float32)
        for i, l in enumerate(batch_seq_len):
            batch_seq_mask[i, :l] = 1
        feed_dict = {img: batch_img, seq_in: batch_seq_in, seq_len: batch_seq_len,
                     seq_mask: batch_seq_mask, seq_truth: batch_seq_truth, init_state_c: zero_state}
        # feed_dict.update(tl.utils.dict_to_one(dropout_dict))
        feed_dict.update(dropout_dict)
        _, batch_loss = sess.run([train_op, loss], feed_dict=feed_dict)
        accum_loss += batch_loss
        if (batch_per_epoch * i_epoch + i_batch + 1) % 100 == 0:
            print('[TRAIN]', 'E=%d/%d'%(i_epoch+1, max_epoch), 'B=%d/%d'%(i_batch+1, batch_per_epoch), 'loss=%f'%(accum_loss/100))
            accum_loss = 0

    bleu_sum = [0] * 4
    rouge_l_sum = 0
    cider_d_sum = 0
    for i_batch in range(n_batch_val):
        batch_img = val_img[i_batch * batch_size:(i_batch + 1) * batch_size]
        seqs, texts = generate_seq(batch_img, k_beam=k_beam)
        for i, seq in enumerate(seqs):
            idx = i_batch * batch_size + i + 8001
            if idx % 20 == 3:
                print('[ VAL ]', idx, texts[i])
            for n in range(1, 5):
                bleu_sum[n - 1] += bleu(seq, val_dict[idx], n)
            rouge_l_sum += rouge_l(seq, val_dict[idx])
            cider_d_sum += cider_d(seq, val_dict[idx], idf, n_val)
    print('[ VAL ]', 'EPOCH %d/%d'%(i_epoch+1, max_epoch))
    for n in range(1, 5):
        print('[ VAL ]', 'BLEU-%d=%f'%(n, bleu_sum[n - 1] / n_val))
    print('[ VAL ]', 'ROUGE-L=%f'%(rouge_l_sum / n_val))
    print('[ VAL ]', 'CIDEr-D=%f'%(cider_d_sum / n_val))

    with open('./results/epoch-%d.txt'%(i_epoch + 1), 'w') as f:
        for i_batch in range(n_batch_test):
            batch_img = test_img[i_batch * batch_size:(i_batch + 1) * batch_size]
            _, texts = generate_seq(batch_img, k_beam=k_beam)
            for s in texts:
                print(s, file=f)