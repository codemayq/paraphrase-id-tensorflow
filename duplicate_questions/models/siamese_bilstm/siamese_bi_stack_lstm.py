from copy import deepcopy
import logging
from overrides import overrides
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
# from tensorflow.metrics import recall,precision,accuracy
from ..base_tf_model import BaseTFModel
from ...util.switchable_dropout_wrapper import SwitchableDropoutWrapper
from ...util.pooling import mean_pool
from ...util.rnn import last_relevant_output
from ...util.rnn import stacked_bidirectional_rnn
logger = logging.getLogger(__name__)


class SiameseBiStackLSTM(BaseTFModel):
    """
    Create a model based off of "Siamese Recurrent Architectures for Learning
    Sentence Similarity" at AAAI '16. The model is super simple: just encode
    both sentences with a LSTM, and then use the function
    exp(-||sentence_one - sentence_two||) to get a probability that the
    two sentences are semantically identical.

    Parameters
    ----------
    mode: str
        One of [train|predict], to indicate what you want the model to do.
        If you pick "predict", then you must also supply the path to a
        pretrained model and DataIndexer to load to the ``predict`` method.

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

    rnn_output_mode: str
        How to calculate the final sentence representation from the RNN
        outputs. mean pool" indicates that the outputs will be averaged (with
        respect to padding), and "last" indicates that the last
        relevant output will be used as the sentence representation.

    output_keep_prob: float
        The probability of keeping an RNN outputs to keep, as opposed
        to dropping it out.
    """

    @overrides
    def __init__(self, config_dict):
        config_dict = deepcopy(config_dict)
        mode = config_dict.pop("mode")
        super(SiameseBiStackLSTM, self).__init__(mode=mode)

        self.word_vocab_size = config_dict.pop("word_vocab_size")
        self.word_embedding_dim = config_dict.pop("word_embedding_dim")
        self.word_embedding_matrix = config_dict.pop("word_embedding_matrix", None)
        self.fine_tune_embeddings = config_dict.pop("fine_tune_embeddings")
        self.rnn_hidden_size = config_dict.pop("rnn_hidden_size")
        self.share_encoder_weights = config_dict.pop("share_encoder_weights")
        self.rnn_output_mode = config_dict.pop("rnn_output_mode")
        self.output_keep_prob = config_dict.pop("output_keep_prob")
        self.num_sentence_words = config_dict.pop("num_sentence_words")
        # TODO num_classes
        self.num_classes = 2
        self.batch_size = config_dict.pop("batch_size")
        if config_dict:
            logger.warning("UNUSED VALUES IN CONFIG DICT: {}".format(config_dict))

    @overrides
    def _create_placeholders(self):
        """
        Create the placeholders for use in the model.
        """
        # Define the inputs here
        # Shape: (batch_size, num_sentence_words)
        # The first input sentence.
        self.sentence_one = tf.placeholder("int32",
                                           [None, self.num_sentence_words],
                                           name="sentence_one")

        # Shape: (batch_size, num_sentence_words)
        # The second input sentence.
        self.sentence_two = tf.placeholder("int32",
                                           [None, self.num_sentence_words],
                                           name="sentence_two")

        # Shape: (batch_size, 2)
        # The true labels, encoded as a one-hot vector. So
        # [1, 0] indicates not duplicate, [0, 1] indicates duplicate.
        self.y_true = tf.placeholder("int32",
                                     [None, self.num_classes],
                                     name="true_labels")

        # A boolean that encodes whether we are training or evaluating
        self.is_train = tf.placeholder('bool', [], name='is_train')

    def contrastive_loss(self, y, d, batch_size):
        tmp = y * tf.square(d)
        # tmp= tf.mul(y,tf.square(d))
        tmp2 = (1 - y) * tf.square(tf.maximum((1 - d), 0))
        return tf.reduce_sum(tmp + tmp2) / batch_size / 2


    @overrides
    def _build_forward(self):
        """
        Using the values in the config passed to the SiameseBiLSTM object
        on creation, build the forward pass of the computation graph.
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
                    # Load the word embedding matrix that was passed in
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
                # Shape: (batch_size, num_sentence_words, embedding_dim)
                word_embedded_sentence_two = tf.nn.embedding_lookup(
                    word_emb_mat,
                    self.sentence_two)

        # Create a convolution + maxpool layer for each filter size
        with tf.name_scope("output"):

            batch_size = tf.shape(self.y_true)[0]
            outputs_one = stacked_bidirectional_rnn(self.rnn_hidden_size,3,word_embedded_sentence_one,sentence_one_len,
                                                    batch_size,self.is_train,self.output_keep_prob)
            outputs_two = stacked_bidirectional_rnn(self.rnn_hidden_size,3,word_embedded_sentence_two,sentence_two_len,
                                                    batch_size,self.is_train,self.output_keep_prob,reuse=True)

            self.out1 = mean_pool(outputs_one,
                                             sentence_one_len)
            self.out2 = mean_pool(outputs_two,
                                             sentence_one_len)

            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.out1, self.out2)), 1, keep_dims=True))
            self.distance = tf.div(self.distance,
                                   tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.out1), 1, keep_dims=True)),
                                          tf.sqrt(tf.reduce_sum(tf.square(self.out2), 1, keep_dims=True))))
            self.distance = tf.reshape(self.distance, [-1], name="distance")
            # self.diff = tf.abs(tf.subtract(self.out1, self.out2), name='err_l1')
            # self.diff = tf.reduce_sum(self.diff, axis=1)
            # self.sim = tf.clip_by_value(tf.exp(-1.0 * self.diff), 1e-7, 1.0 - 1e-7)

            self.pre_cate = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance),
                                        name="pre_cate")  # auto threshold 0.5

        argmax_true = tf.argmax(self.y_true, 1)
        with tf.name_scope("loss"):
            # Use the exponential of the negative L1 distance
            # between the two encoded sentences to get an output
            # distribution over labels.
            # Shape: (batch_size, 2)

            # Manually calculating cross-entropy, since we output
            # probabilities and can't use softmax_cross_entropy_with_logits
            # Add epsilon to the probabilities in order to prevent log(0)
            # self.loss = tf.reduce_mean(
            #     -tf.reduce_sum(tf.cast(self.y_true, "float") *
            #                    tf.log(self.y_pred),
            #                    axis=1))
            self.y_pred = tf.one_hot(tf.cast(self.pre_cate, "int64"), 2)
            self.loss = self.contrastive_loss(tf.cast(argmax_true, "float"), self.distance,  tf.cast(batch_size,"float"))

        self.pre_cate = tf.argmax(self.y_pred, 1)
        argmax_pred = tf.argmax(self.y_pred, 1)

        with tf.name_scope("accuracy"):
            # Get the correct predictions.
            # Shape: (batch_size,) of bool

            correct_predictions = tf.equal(self.pre_cate, argmax_true)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

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
            # tf.summary.scalar("acc", self.acc)
            # tf.summary.scalar("precision", self.precision)
            tf.summary.scalar("f1", self.f1)
            self.summary_op = tf.summary.merge_all()

    @overrides
    def _get_train_feed_dict(self, batch):
        inputs, targets, lineids = batch
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

