from copy import deepcopy
import logging
from overrides import overrides
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell

from ..base_tf_model import BaseTFModel
from ...util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from ...util.pooling import mean_pool

logger = logging.getLogger(__name__)


class SiameseMatchingBiLSTM(BaseTFModel):
    """
    Create a model based off of the baseline (no inner-attention) in
    "Learning Natural Language Inference using Bidirectional LSTM model
    and Inner-Attention" (https://arxiv.org/abs/1605.09090).

    The model is super simple: just encode
    both sentences with a LSTM, and take the mean pool over the timesteps
    as the sentence representation. Then, create a vector with the
    by concatenating (||) the following:
    sentence1|sentence1-sentence2|sentence1*sentence2|sentence2

    Lastly, run this vector through a dense layer to (relu activation)
    to get the logits, which are then softmaxed to get a probability
    distribution [is_not_duplicate, is_duplicate].

    The input config is an argarse Namespace storing a variety of configuration
    values that are necessary to build the graph. The keys we expect
    in this Namespace are outlined below.

    Parameters
    ----------
    mode: str
        One of {train|predict}, to indicate what you want the model to do.
        If you pick "predict", then you must also supply the path to a
        pretrained model and DataIndexer to load.

    word_vocab_size: int
        The number of unique tokens in the dataset, plus the UNK and padding
        tokens. Alternatively, the highest index assigned to any word, +1.
        This is used by the model to figure out the dimensionality of the
        embedding matrix.

    word_embedding_dim: int
        The length of a word embedding. This is used by
        the model to figure out the dimensionality of the embedding matrix.

    word_embedding_matrix: numpy array, optional if predicting
        A numpy array of shape (word_vocab_size, word_emb_dim).
        word_embedding_matrix[index] should represent the word vector for
        that particular word index. This is used to initialize the
        word embedding matrix in the model, and is optional if predicting
        since we assume that the word embeddings variable will be loaded
        with the model.

    fine_tune_embeddings: boolean
        If true, sets the embeddings to be trainable.

    rnn_hidden_size: int
        The output dimension of the RNN encoder. Note that this model uses a
        bidirectional LSTM, so the actual sentence vectors will be
        of length 2*rnn_hidden_size.

    share_encoder_weights: boolean
        Whether to use the same encoder on both input sentnces (thus
        sharing weights), or a different one for each sentence.

    output_keep_prob: float
        The probability of keeping an RNN outputs to keep, as opposed
        to dropping it out.
    """
    def __init__(self, config_dict):
        config_dict = deepcopy(config_dict)
        mode = config_dict.pop("mode")
        super(SiameseMatchingBiLSTM, self).__init__(mode=mode)

        self.word_vocab_size = config_dict.pop("word_vocab_size")
        self.word_embedding_dim = config_dict.pop("word_embedding_dim")
        self.word_embedding_matrix = config_dict.pop("word_embedding_matrix", None)
        self.fine_tune_embeddings = config_dict.pop("fine_tune_embeddings")
        self.rnn_hidden_size = config_dict.pop("rnn_hidden_size")
        self.share_encoder_weights = config_dict.pop("share_encoder_weights")
        self.output_keep_prob = config_dict.pop("output_keep_prob")

        self.sequence_length = config_dict.pop("num_sentence_words")
        if config_dict:
            logger.warning("UNUSED VALUES IN CONFIG DICT: {}".format(config_dict))

    def _create_placeholders(self):
        """
        Create the placeholders for use in the model.
        """
        # Define the inputs here
        # Shape: (batch_size, num_sentence_words)
        # The first input sentence.
        self.sentence_one = tf.placeholder("int32",
                                           [None, None],
                                           name="sentence_one")

        # Shape: (batch_size, num_sentence_words)
        # The second input sentence.
        self.sentence_two = tf.placeholder("int32",
                                           [None, None],
                                           name="sentence_two")
        # Shape: (batch_size, 2)
        # The true labels, encoded as a one-hot vector. So
        # [1, 0] indicates not duplicate, [0, 1] indicates duplicate.
        self.y_true = tf.placeholder("int32",
                                     [None, 2],
                                     name="true_labels")

        # A boolean that encodes whether we are training or evaluating
        self.is_train = tf.placeholder('bool', [], name='is_train')

    def _build_forward(self):
        """
        Using the config passed to the SiameseMatchingBiLSTM object on
        creation, build the forward pass of the computation graph.
        """
        # A mask over the word indices in the sentence, indicating
        # which indices are padding and which are words.
        # Shape: (batch_size, num_sentence_words)
        sentence_one_mask = tf.sign(self.sentence_one,
                                    name="sentence_one_masking")
        sentence_two_mask = tf.sign(self.sentence_two,
                                    name="sentence_two_masking")

        # The unpadded lengths of sentence one and sentence two
        # Shape: (batch_size,)
        sentence_one_len = tf.reduce_sum(sentence_one_mask, 1)
        sentence_two_len = tf.reduce_sum(sentence_two_mask, 1)

        word_vocab_size = self.word_vocab_size
        word_embedding_dim = self.word_embedding_dim
        word_embedding_matrix = self.word_embedding_matrix
        fine_tune_embeddings = self.fine_tune_embeddings

        with tf.variable_scope("embeddings"):
            with tf.variable_scope("embedding_var"), tf.device("/cpu:0"):
                if self.mode == "train":
                    # Load the word embedding matrix from the config,
                    # since we are training
                    word_emb_mat = tf.get_variable(
                        "word_emb_mat",
                        dtype="float",
                        shape=[word_vocab_size,
                               word_embedding_dim],
                        initializer=tf.constant_initializer(
                            word_embedding_matrix),
                        trainable=fine_tune_embeddings)
                else:
                    # We are not training, so a model should have been
                    # loaded with the embedding matrix already there.
                    word_emb_mat = tf.get_variable("word_emb_mat",
                                                   shape=[word_vocab_size,
                                                          word_embedding_dim],
                                                   dtype="float",
                                                   trainable=fine_tune_embeddings)

            with tf.variable_scope("word_embeddings"):
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_one = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_one)
                self.word_embedded_sentence_one_expanded = tf.expand_dims(word_embedded_sentence_one, -1)
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_two = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_two)
                self.word_embedded_sentence_one_expanded = tf.expand_dims(word_embedded_sentence_two, -1)

        num_filters = 256
        filter_sizes = [1,2,3,4,5,6]
        embedding_size = 300


        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.word_embedded_sentence_one_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")

                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.output_keep_prob)

        l2_loss = tf.constant(0.0)
        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, 2],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")


        rnn_hidden_size = self.rnn_hidden_size
        output_keep_prob = self.output_keep_prob







        # Sentence matching layer
        with tf.name_scope("match_sentences"):
            sentence_difference = encoded_sentence_one - encoded_sentence_two
            sentence_product = encoded_sentence_one * encoded_sentence_two
            # Shape: (batch_size, 4 * 2*rnn_hidden_size)
            matching_vector = tf.concat([encoded_sentence_one, sentence_product,
                                         sentence_difference, encoded_sentence_two], 1)
        # Nonlinear projection to 2 dimensional class probabilities
        with tf.variable_scope("project_matching_vector"):
            # Shape: (batch_size, 2)
            projection = tf.layers.dense(matching_vector, 2, tf.nn.relu,
                                         name="matching_vector_projection")

        with tf.name_scope("loss"):
            # Get the predicted class probabilities
            # Shape: (batch_size, 2)
            self.y_pred = tf.nn.softmax(projection, name="softmax_probabilities")
            # Use softmax_cross_entropy_with_logits to calculate xentropy.
            # It's unideal to do the softmax twice, but I prefer the numerical
            # stability of the tf function.

            class_weights = tf.constant([[1.0, 2.0]])
            weights = tf.reduce_sum(class_weights * tf.cast(self.y_true, "float"), axis=1)
            unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_true, logits=projection)
            weighted_losses = unweighted_losses * weights
            self.loss = tf.reduce_mean(weighted_losses)

        self.pre_cate = tf.argmax(self.y_pred, 1)
        argmax_pred = tf.argmax(self.y_pred, 1)
        argmax_true = tf.argmax(self.y_true, 1)

        with tf.name_scope("accuracy"):
            # Get the correct predictions.
            # Shape: (batch_size,) of bool
            correct_predictions = tf.equal(
                argmax_pred,
                argmax_true)

            # Cast to float, and take the mean to get accuracy
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
                                                   "float"))

        TP = tf.count_nonzero(argmax_pred * argmax_true, dtype=tf.float32)
        TN = tf.count_nonzero((argmax_pred - 1) * (argmax_true - 1), dtype=tf.float32)
        FP = tf.count_nonzero(argmax_pred * (argmax_true - 1), dtype=tf.float32)
        FN = tf.count_nonzero((argmax_pred - 1) * argmax_true, dtype=tf.float32)

        # with tf.name_scope("acc"):
        #     self.acc = accuracy(argmax_true, argmax_pred)

        # with tf.name_scope("precision"):
        #     self.precision = precision(argmax_true, argmax_pred)
        #
        # with tf.name_scope("recall"):
        #     self.recall = recall(argmax_true, argmax_pred)

        with tf.name_scope("f1"):
            # self.f1 = 2 * precision * recall / (precision + recall)
            precision = TP / (TP + FP + 1)
            recall = TP / (TP + FN + 1)
            self.f1 = 2 * precision * recall / (precision + recall + 1e-08)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer()
            self.training_op = optimizer.minimize(self.loss,
                                                  global_step=self.global_step)

        with tf.name_scope("train_summaries"):
            # Add the loss and the accuracy to the tensorboard summary
            tf.summary.scalar("loss", self.loss)
            tf.summary.scalar("accuracy", self.accuracy)
            tf.summary.scalar("f1", self.f1)

            self.summary_op = tf.summary.merge_all()

    @overrides
    def _get_train_feed_dict(self, batch):
        inputs, targets, lineids  = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.y_true: targets[0],
                     self.is_train: True}
        return feed_dict

    @overrides
    def _get_validation_feed_dict(self, batch):
        inputs, targets, lineids = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.y_true: targets[0],
                     self.is_train: False}
        return feed_dict

    @overrides
    def _get_test_feed_dict(self, batch):
        inputs, _, lineids = batch
        feed_dict = {self.sentence_one: inputs[0],
                     self.sentence_two: inputs[1],
                     self.is_train: False}
        return feed_dict
