# coding=UTF-8
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from ..util.switchable_dropout_wrapper import SwitchableDropoutWrapper
def last_relevant_output(output, sequence_length):
    """
    Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).

    Parameters
    ----------
    output: Tensor
        A tensor, generally the output of a tensorflow RNN.
        The tensor index sequence_lengths+1 is selected for each
        instance in the output.

    sequence_length: Tensor
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    last_relevant_output: Tensor
        The last relevant output (last element of the sequence), as retrieved
        by the output Tensor and indicated by the sequence_length Tensor.
    """
    with tf.name_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[-2]
        out_size = int(output.get_shape()[-1])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


def stacked_bidirectional_rnn(num_units, num_layers, inputs, seq_lengths, batch_size ,is_train, output_keep_prob, reuse=False):

    """

    multi layer bidirectional rnn

    :param num_units: int, hidden unit of RNN cell

    :param num_layers: int, the number of layers

    :param inputs: Tensor, the input sequence, shape: [batch_size, max_time_step, num_feature]

    :param seq_lengths: list or 1-D Tensor, sequence length, a list of sequence lengths, the length of the list is batch_size

    :param batch_size: int

    :return: the output of last layer bidirectional rnn with concatenating

    这里用到几个tf的特性

    1. tf.variable_scope(None, default_name="bidirectional-rnn")使用default_name

    的话,tf会自动处理命名冲突

    """

    # TODO: add time_major parameter, and using batch_size = tf.shape(inputs)[0], and more assert

    _inputs = inputs

    if len(_inputs.get_shape().as_list()) != 3:
        raise ValueError("the inputs must be 3-dimentional Tensor")

    for i in range(num_layers):
        with tf.variable_scope("Layer%d" % i, reuse=reuse):
            rnn_cell_fw = LSTMCell(num_units)
            rnn_cell_bw = LSTMCell(num_units)
            rnn_cell_fw = SwitchableDropoutWrapper(rnn_cell_fw, is_train,
                                                   output_keep_prob=output_keep_prob)

            rnn_cell_bw = SwitchableDropoutWrapper(rnn_cell_bw, is_train,
                                                   output_keep_prob=output_keep_prob)
            initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)

            initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)

            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,

                                                              initial_state_fw, initial_state_bw, dtype=tf.float32)
            _inputs = tf.concat(output, 2)

    return _inputs
