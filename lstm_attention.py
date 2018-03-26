import tensorflow as tf
import tensorlayer as tl
import numpy as np
from datetime import datetime
import os

from utils.load_data import *
from utils.text_process import *
from utils.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('n_epoch', 10, 'Number of epoches to train')
flags.DEFINE_integer('n_hidden', 512, 'Dimension of hidden states in LSTM')
flags.DEFINE_integer('n_embed', 512, 'Dimension of word embedding vectors')
flags.DEFINE_integer('k_beam', 3, 'Width of beam in beam search')
flags.DEFINE_float('lam', 0.1, 'Regularization factor on alpha (attention strength)')
flags.DEFINE_float('end_penalty', 1.0, 'Penalty of each timestep after <END> in beam search')
flags.DEFINE_boolean('use_partial_val', False, 'Set this only when comparing results with validation set agreement')
flags.DEFINE_boolean('print_val', True, 'Print generated captions for selected images of validation set in log')
flags.DEFINE_boolean('output_test', True, 'Output generated captions for test set')
flags.DEFINE_string('output_dir', datetime.now().strftime('%y%m%d%H%M'), 'Name of output directory')

d_local = 512
a_local = 7
n_epoch = FLAGS.n_epoch
n_hidden = FLAGS.n_hidden
n_embed = FLAGS.n_embed
keep_prob = 0.5
k_beam = FLAGS.k_beam
lam = FLAGS.lam
end_penalty = FLAGS.end_penalty
use_partial_val = FLAGS.use_partial_val
print_val = FLAGS.print_val
output_test = FLAGS.output_test
output_dir = FLAGS.output_dir
try:
    os.mkdir(os.path.join('results', output_dir))
except FileExistsError:
    pass

train_img, val_img, test_img = load_img('./data/image_vgg19_block5_pool_feature.h5')
n_val = val_img.shape[0]
n_test = test_img.shape[0]
train_idx, train_sentences = load_text('./data/train.txt')
train_dict = {}
for idx, stc in zip(train_idx, train_sentences):
    train_dict.setdefault(idx, []).append(stc)
for idx in train_dict.keys():
    if len(train_dict[idx]) < 5:
        orig = train_dict[idx][:]
        shuffle_idx = np.random.permutation(len(orig))
        for i in range(5 - len(orig)):
            train_dict[idx].append(orig[shuffle_idx[i % len(orig)]])
train_pairs = [(idx, stc) for idx, stcs in train_dict.items() for stc in stcs]
train_idx = np.array([idx for idx, stc in train_pairs])
train_sentences = [stc for idx, stc in train_pairs]
train_seq, vocab, vocab_inv = encode_text(train_sentences)

val_idx, val_sentences = load_text('./data/valid.txt')
val_seq = encode_text(val_sentences, vocab=vocab, ignore_non_chinese=False, with_begin_end=False)
val_dict = {}
val_test = {}
for idx, seq in zip(val_idx, val_seq):
    if val_dict.get(idx) is None:
        if use_partial_val:
            val_test[idx] = seq
            val_dict[idx] = []
        else:
            val_dict[idx] = [seq]
    else:
        val_dict[idx].append(seq)
for idx in val_dict.keys():
    if len(val_dict[idx]) == 0:
        val_dict[idx] = [val_test[idx]]

idf = build_idf(val_dict)


train_seq, train_len = seq2array(train_seq)
max_step = train_seq.shape[1]
n_train = len(train_idx)

sess = tf.InteractiveSession()

img_local = tf.placeholder(tf.float32, shape=[None, a_local, a_local, d_local], name='img')
seq_in = tf.placeholder(tf.int64, shape=[None, None], name='seq_in')
seq_len = tf.placeholder(tf.int64, shape=[None], name='seq_len')
seq_max_len = tf.shape(seq_in)[1]
seq_truth = tf.placeholder(tf.int64, shape=[None, None], name='seq_truth')

class FixedAttentionLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, attn_target):
        super(FixedAttentionLSTMCell, self).__init__()
        self._cell = tf.contrib.rnn.BasicLSTMCell(num_units=num_units)
        self.target = attn_target
        _, self.n_target, self.target_size = attn_target.get_shape().as_list()
        self.num_units = num_units

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units + self.target_size + self.n_target

    def __call__(self, inputs, state):
        state_h = state.h
        with tf.variable_scope('attention'):
            Wh = tf.get_variable('Wh', [self.num_units, self.target_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            Wa = tf.get_variable('Wa', [self.target_size, self.target_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            M = tf.tanh(
                tf.reshape(tf.matmul(tf.reshape(self.target, [-1, self.target_size]), Wa), [-1, self.n_target, self.target_size])
                + tf.expand_dims(tf.matmul(state_h, Wh), axis=1))
            w_alpha = tf.get_variable('w_alpha', [self.target_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
            alpha = tf.nn.softmax(tf.reduce_sum(M * tf.reshape(w_alpha, [1, 1, -1]), axis=2, keep_dims=True), dim=1)
            w_beta = tf.get_variable('w_beta', [self.num_units], initializer=tf.truncated_normal_initializer(stddev=0.1))
            b_beta = tf.get_variable('b_beta', [], initializer=tf.constant_initializer(0))
            beta = tf.sigmoid(tf.matmul(state_h, tf.expand_dims(w_beta, axis=1)) + b_beta)
            attn_out = tf.reduce_sum(self.target * alpha, axis=1)
        cell_input = tf.concat([inputs, attn_out * beta], axis=1)
        output, new_state = self._cell(cell_input, state)
        new_output = tf.concat([output, attn_out, tf.squeeze(alpha)], axis=1)
        return new_output, new_state

attn_net = tl.layers.InputLayer(tf.reshape(img_local, [-1, d_local]), name='attn_in')
attn_net = tl.layers.DenseLayer(attn_net,
    n_units=d_local,
    act=tf.sigmoid,
    name='attn_proj')
attn_net = tl.layers.ReshapeLayer(attn_net, shape=[-1, a_local ** 2, d_local], name='attn_reshape')
attn_target = attn_net.outputs
target_mean = tf.reduce_mean(attn_target, axis=1)

init_net = tl.layers.InputLayer(inputs=target_mean, name='init_in')
init_net = tl.layers.DenseLayer(init_net,
    n_units=n_hidden * 2,
    act=tf.sigmoid,
    name='init')
init_c, init_h = tf.split(init_net.outputs, 2, axis=1)

lstm_net = tl.layers.EmbeddingInputlayer(
    inputs=seq_in,
    vocabulary_size=len(vocab),
    embedding_size=n_embed,
    name='embedding')
embedded_input = lstm_net.outputs
lstm_net = tl.layers.DropoutLayer(lstm_net, keep=keep_prob, name='lstm_in_dropout')
init_state_c = tf.placeholder_with_default(init_c, shape=[None, n_hidden], name='init_state_c')
init_state_h = tf.placeholder_with_default(init_h, shape=[None, n_hidden], name='init_state_h')
lstm_net = tl.layers.DynamicRNNLayer(lstm_net,
    cell_fn=FixedAttentionLSTMCell,
    n_hidden=n_hidden,
    cell_init_args={'attn_target': attn_target},
    sequence_length=seq_len,
    initial_state=tf.contrib.rnn.LSTMStateTuple(init_state_c, init_state_h),
    name='lstm')
state_outputs = lstm_net.final_state
out, attn, alpha = tf.split(lstm_net.outputs, [n_hidden, d_local, a_local ** 2], axis=2)

output_net = tl.layers.InputLayer(tf.concat([embedded_input, attn], axis=2), name='out1_in')
output_net = tl.layers.ReshapeLayer(output_net, shape=[-1, n_embed + d_local], name='out1_in_reshape')
output_net = tl.layers.DenseLayer(output_net,
    n_units=n_hidden,
    act=tf.tanh,
    name='out1')
output_net = tl.layers.ReshapeLayer(output_net, shape=[-1, seq_max_len, n_hidden], name='out1_out_reshape')
out_input = tl.layers.InputLayer(out, name='out2_in')
output_net = tl.layers.ElementwiseLayer([output_net, out_input], combine_fn=tf.add, name='out_merge')
output_net = tl.layers.DropoutLayer(output_net, keep=keep_prob, name='out2_dropout')
output_net = tl.layers.ReshapeLayer(output_net, shape=[-1, n_hidden], name='output2_in_reshape')
output_net = tl.layers.DenseLayer(output_net,
    n_units=len(vocab),
    act=tf.identity,
    name='output_unembed')
output_net = tl.layers.ReshapeLayer(output_net, shape=[-1, seq_max_len, len(vocab)], name='output2_out_reshape')

k_top = tf.placeholder_with_default(1, shape=[], name='k_top')
top_k_loglike, top_k_ind = tf.nn.top_k(tf.nn.log_softmax(output_net.outputs), k=k_top)
loss_per_word = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seq_truth, logits=output_net.outputs, name='cross_entropy')
seq_mask = tf.sequence_mask(seq_len, max_step - 1, dtype=tf.float32)
p_loss = tf.reduce_sum(loss_per_word * seq_mask)
reg_loss = tf.reduce_sum(tf.square(tf.reduce_sum(alpha * tf.expand_dims(seq_mask, axis=2), axis=1) - 1))
loss = p_loss + lam * reg_loss

dropout_dict = {**attn_net.all_drop, **init_net.all_drop, **lstm_net.all_drop, **output_net.all_drop}
train_vars = attn_net.all_params + init_net.all_params + lstm_net.all_params + output_net.all_params
train_op = tf.train.AdamOptimizer().minimize(loss, var_list=train_vars)


def generate_seq(batch_img, k_beam=1):
    batch_size = batch_img.shape[0]
    first_dim = batch_size * k_beam
    batch_img_in = np.repeat(batch_img, k_beam, axis=0)
    batch_seq_in = np.zeros((first_dim, 1), dtype=np.int64)
    batch_seq_len = np.ones((first_dim,), dtype=np.int64)
    batch_res = np.zeros((first_dim, max_step - 1), dtype=np.int64)
    batch_loglike_sum = np.zeros((first_dim,), dtype=np.float32)
    for t in range(max_step - 1):
        if t == 0:
            batch_seq_in[:, 0] = vocab['<BEG>']
            feed_dict = {img_local: batch_img_in, k_top: k_beam,
                         seq_in: batch_seq_in, seq_len: batch_seq_len}
        else:
            batch_seq_in[:, 0] = new_step
            feed_dict = {img_local: batch_img_in, k_top: k_beam,
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
                        step_loglike[i * k_beam + j, 0, 0] = -end_penalty
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
            break
    best_res = batch_res[::k_beam, :]
    seqs = array2seq(best_res, vocab['<END>'])
    texts = decode_text(seqs, vocab_inv)
    return seqs, texts

sess.run(tf.global_variables_initializer())

batch_size = 20
batch_per_epoch = n_train // batch_size
# batch_per_epoch = 100
n_batch_val = n_val // batch_size
n_batch_test = n_test // batch_size
accum_p_loss = 0
accum_reg_loss = 0
shuffle_idx = np.random.permutation(n_train)
for i_epoch in range(n_epoch):
    for i_batch in range(batch_per_epoch):
        batch_idx = shuffle_idx[i_batch * batch_size:(i_batch + 1) * batch_size]
        batch_img = train_img[train_idx[batch_idx] - 1, :]
        batch_seq_in = train_seq[batch_idx, :-1]
        batch_seq_truth = train_seq[batch_idx, 1:]
        batch_seq_len = train_len[batch_idx] - 1
        feed_dict = {img_local: batch_img, seq_in: batch_seq_in, seq_len: batch_seq_len, seq_truth: batch_seq_truth}
        feed_dict.update(dropout_dict)
        _, batch_p_loss, batch_reg_loss = sess.run([train_op, p_loss, reg_loss], feed_dict=feed_dict)
        accum_p_loss += batch_p_loss
        accum_reg_loss += batch_reg_loss
        if (batch_per_epoch * i_epoch + i_batch + 1) % 100 == 0:
            print('[TRAIN]', 'E=%d/%d'%(i_epoch+1, n_epoch), 'B=%d/%d'%(i_batch+1, batch_per_epoch),
                  'p_loss=%f'%(accum_p_loss / 100), 'reg_loss=%f'%(accum_reg_loss / 100))
            accum_p_loss = 0
            accum_reg_loss = 0

    bleu_counter = BLEUCounter()
    rouge_l_sum = 0
    cider_d_sum = 0
    val_file_path = os.path.join('results', output_dir, 'val_meteor_E%02d.txt'%(i_epoch + 1))
    with open(val_file_path, 'w', encoding='utf8') as f:
        for i_batch in range(n_batch_val):
            batch_img = val_img[i_batch * batch_size:(i_batch + 1) * batch_size]
            seqs, texts = generate_seq(batch_img, k_beam=k_beam)
            text_seged = decode_text(seqs, vocab_inv, sep=' ')
            for i, seq in enumerate(seqs):
                idx = i_batch * batch_size + i + 8001
                print(text_seged[i], file=f)
                if idx % 20 == 3 and print_val:
                    print('[ VAL ]', idx, texts[i])
                bleu_counter.add(seq, val_dict[idx])
                rouge_l_sum += rouge_l(seq, val_dict[idx])
                cider_d_sum += cider_d(seq, val_dict[idx], idf, n_val)
    print('[ VAL ]', 'EPOCH %d/%d'%(i_epoch+1, n_epoch))
    bleus = bleu_counter.get_bleu()
    for n in range(1, 5):
        print('[ VAL ]', 'BLEU-%d=%f'%(n, bleus[n - 1]))
    print('[ VAL ]', 'METEOR=%f'%meteor(val_file_path, 'data/valid_meteor.txt'))
    print('[ VAL ]', 'ROUGE-L=%f'%(rouge_l_sum / n_val))
    print('[ VAL ]', 'CIDEr-D=%f'%(cider_d_sum / n_val))

    if output_test:
        with open(os.path.join('results', output_dir, 'test_E%02d.txt'%(i_epoch + 1)), 'w', encoding='utf8') as f:
            for i_batch in range(n_batch_test):
                batch_img = test_img[i_batch * batch_size:(i_batch + 1) * batch_size]
                _, texts = generate_seq(batch_img, k_beam=k_beam)
                for s in texts:
                    print(s, file=f)
