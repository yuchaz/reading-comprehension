import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper
from tensorflow.contrib.rnn import GRUCell, LSTMCell


def cbow_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):

        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        # emb_mat = tf.get_variable('emb_mat', shape=[V, d])
        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        qq_avg = tf.reduce_mean(bool_mask(qq, q_mask, expand=True), axis=1)  # [N, d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1)  # [N, 1, d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1])  # [N, JX, d]

        xq = tf.concat([xx, qq_avg_tiled, xx * qq_avg_tiled], axis=2)  # [N, JX, 3d]
        xq_flat = tf.reshape(xq, [-1, 3*d])  # [N * JX, 3*d]

        # Compute logits
        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def rnn_forward(config, inputs, scope=None):
    cell_fw = cell_bw = DropoutWrapper(GRUCell(config.hidden_size),
        input_keep_prob=config.keep_prob, output_keep_prob=config.keep_prob)
    return bidirectional_rnn(config, inputs, cell_fw, cell_bw, scope)
def lstm_forward(config, inputs, scope=None):
    cell_fw = cell_bw = DropoutWrapper(LSTMCell(config.hidden_size),
        input_keep_prob=config.keep_prob, output_keep_prob=config.keep_prob)
    return bidirectional_rnn(config, inputs, cell_fw, cell_bw, scope)
def lstm_fw_gru_bw(config, inputs, scope=None):
    cell_fw = DropoutWrapper(LSTMCell(config.hidden_size),
        input_keep_prob=config.keep_prob, output_keep_prob=config.keep_prob)
    cell_bw = DropoutWrapper(GRUCell(config.hidden_size),
        input_keep_prob=config.keep_prob, output_keep_prob=config.keep_prob)
    return bidirectional_rnn(config, inputs, cell_fw, cell_bw, scope)
def gru_fw_lstm_bw(config, inputs, scope=None):
    cell_fw = DropoutWrapper(GRUCell(config.hidden_size),
        input_keep_prob=config.keep_prob, output_keep_prob=config.keep_prob)
    cell_bw = DropoutWrapper(LSTMCell(config.hidden_size),
        input_keep_prob=config.keep_prob, output_keep_prob=config.keep_prob)
    return bidirectional_rnn(config, inputs, cell_fw, cell_bw, scope)

def bidirectional_rnn(config, inputs, cell_fw, cell_bw, scope=None):
    with tf.variable_scope(scope or "forward"):
        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        q_inputs = bool_mask(qq, q_mask, expand=True)
        x_inputs = bool_mask(xx, x_mask, expand=True)
        q_outputs, qo_ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=q_inputs,
            sequence_length=q_len, dtype=tf.float32,
            scope="q_bidirectional_rnn") # q_outputs = [N, JQ, d]
        qo = tf.concat(list(q_outputs), axis=2) # [N, JQ, 2d]
        x_outputs, xo_ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell_fw, cell_bw=cell_bw, inputs=x_inputs,
            sequence_length=x_len, dtype=tf.float32,
            scope="x_bidirectional_rnn") # x_outputs = [N, JX, d]
        xo = tf.concat(list(x_outputs), axis=2) # [N, JX, 2d]
        qq_avg = tf.reduce_mean(qo,axis=1) # [N, 2d]
        qq_avg_exp = tf.expand_dims(qq_avg, axis=1) # [N, 1, 2d]
        qq_avg_tiled = tf.tile(qq_avg_exp, [1, JX, 1]) # [N, JX, 2d]

        xq = tf.concat([xo, qq_avg_tiled, xo * qq_avg_tiled], axis=2)  # [N, JX, 6d]
        xq_flat = tf.reshape(xq, [-1, 6*d])  # [N * JX, 6*d]

        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs

def attention_forward(config, inputs, scope=None):
    with tf.variable_scope(scope or "forward"):
        JX, JQ = config.max_context_size, config.max_ques_size
        d = config.hidden_size
        x, x_len, q, q_len = [inputs[key] for key in ['x', 'x_len', 'q', 'q_len']]
        x_mask = tf.sequence_mask(x_len, JX)
        q_mask = tf.sequence_mask(q_len, JQ)

        emb_mat = config.emb_mat_ph if config.serve else config.emb_mat
        emb_mat = tf.slice(emb_mat, [2, 0], [-1, -1])
        emb_mat = tf.concat([tf.get_variable('emb_mat', shape=[2, d]), emb_mat], axis=0)
        xx = tf.nn.embedding_lookup(emb_mat, x, name='xx')  # [N, JX, d]
        qq = tf.nn.embedding_lookup(emb_mat, q, name='qq')  # [N, JQ, d]

        xx_extend = tf.expand_dims(xx, axis=2) # [N, JX, 1, d]
        xx_tiled = tf.tile(xx_extend, [1, 1, JQ, 1]) # [N, JX, JQ, d]
        qq_extend = tf.expand_dims(qq, axis=1) # [N, 1, JQ, d]
        qq_tiled = tf.tile(qq_extend, [1, JX, 1, 1]) # [N, JX, JQ, d]

        xq_attention = tf.einsum('ijl,ikl->ijkl', xx, qq) # [N, JX, JQ, d]
        attention = tf.concat([xx_tiled, qq_tiled, xq_attention], axis=3) # [N, JX, JQ, 3d]
        W_attention = tf.Variable(
            tf.random_uniform([1, 3*d], -1.0, 1.0),
            name="W_attention")
        b_attention = tf.Variable(tf.constant(0.1, shape=[1]), name="b_attention")
        p = tf.einsum('ijkl,lm->ijkm', attention,W_attention) + b_attention # [N, JX, JQ]

        cell = DropoutWrapper(
            GRUCell(d), input_keep_prob=config.keep_prob, output_keep_prob=config.keep_prob)

        q_inputs = bool_mask(qq, q_mask, expand=True)
        x_inputs = bool_mask(xx, x_mask, expand=True)
        q_outputs, qo_ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell, cell_bw=cell, inputs=q_inputs,
            sequence_length=q_len, dtype=tf.float32,
            scope="q_bidirectional_rnn") # q_outputs = [N, JQ, d]
        qo = tf.concat(list(q_outputs), axis=2) # [N, JQ, 2d]
        x_outputs, xo_ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell, cell_bw=cell, inputs=x_inputs,
            sequence_length=x_len, dtype=tf.float32,
            scope="x_bidirectional_rnn") # x_outputs = [N, JX, d]
        xo = tf.concat(list(x_outputs), axis=2) # [N, JX, 2d]

        qq_weight_avg = tf.einsum('ijk,ikl->ijl', p, qo) # [N, JX, 2d]

        xq = tf.concat([xo, qq_weight_avg, xo * qq_weight_avg], axis=2)  # [N, JX, 6d]
        xq_flat = tf.reshape(xq, [-1, 6*d])  # [N * JX, 6*d]

        with tf.variable_scope('start'):
            logits1 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp1 = tf.argmax(logits1, axis=1)  # [N]
        with tf.variable_scope('stop'):
            logits2 = exp_mask(tf.reshape(tf.layers.dense(inputs=xq_flat, units=1), [-1, JX]), x_mask)  # [N, JX]
            yp2 = tf.argmax(logits2, axis=1)  # [N]

        outputs = {'logits1': logits1, 'logits2': logits2, 'yp1': yp1, 'yp2': yp2}
        variables = {'emb_mat': emb_mat}
        return variables, outputs


def get_loss(config, inputs, outputs, scope=None):
    with tf.name_scope(scope or "loss"):
        y1, y2 = inputs['y1'], inputs['y2']
        logits1, logits2 = outputs['logits1'], outputs['logits2']
        loss1 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y1, logits=logits1))
        loss2 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y2, logits=logits2))
        loss = loss1 + loss2
        acc1 = tf.reduce_mean(tf.cast(tf.equal(y1, tf.cast(tf.argmax(logits1, 1), 'int32')), 'float'))
        acc2 = tf.reduce_mean(tf.cast(tf.equal(y2, tf.cast(tf.argmax(logits2, 1), 'int32')), 'float'))
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('acc1', acc1)
        tf.summary.scalar('acc2', acc2)
        return loss


def exp_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val - (1.0 - tf.cast(mask, 'float')) * 10.0e10


def bool_mask(val, mask, expand=False):
    if expand:
        mask = tf.expand_dims(mask, -1)
    return val * tf.cast(mask, 'float')
