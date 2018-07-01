import csv
from copy import deepcopy
import itertools
import numpy as np
from overrides import overrides

from .instance import TextInstance, IndexedInstance
from .sts_instance import STSInstance,IndexedSTSInstance
from .instance_word import IndexedInstanceWord
from ..tokenizers.word_tokenizers import ChineseWordTokenizer


class WeizhongInstance(STSInstance):
    def __init__(self, first_sentence, second_sentence, label, line_id, tokenizer=ChineseWordTokenizer):
        # self.tokenizer = ChineseWordTokenizer()
        super(WeizhongInstance, self).__init__(first_sentence, second_sentence, label, tokenizer)
        self.line_id = line_id

    @classmethod
    def read_from_line(cls, line):
        fields = line.strip().split("\t")
        if len(fields) == 3:
            if fields[0].isdigit():
                # test set instance
                line_id, first_sentence, second_sentence = fields
                label = None
                line_id = int(line_id)
            else:
                 # train set instance
                first_sentence, second_sentence, label = fields
                label = int(label)
                line_id = 0
        else:
            raise RuntimeError("Unrecognized line format: " + line)
        return cls(first_sentence, second_sentence, label, line_id)

    @overrides
    def to_indexed_instance(self, data_indexer):
        indexed_first_words, indexed_first_chars = self._index_text(
            self.first_sentence_tokenized,
            data_indexer)
        indexed_second_words, indexed_second_chars = self._index_text(
            self.second_sentence_tokenized,
            data_indexer)
        # These are lists of IndexedInstanceWords
        indexed_first_sentence = [IndexedInstanceWord(word, word_characters) for
                                  word, word_characters in zip(indexed_first_words,
                                                               indexed_first_chars)]
        indexed_second_sentence = [IndexedInstanceWord(word, word_characters) for
                                   word, word_characters in zip(indexed_second_words,
                                                                indexed_second_chars)]
        return WeizhongIndexInstance(indexed_first_sentence,
                                     indexed_second_sentence,
                                     self.label_mapping[self.label], self.line_id)


class WeizhongIndexInstance(IndexedSTSInstance):
    def __init__(self, first_sentence_indices, second_sentence_indices, label, line_id):
        super(WeizhongIndexInstance, self).__init__(first_sentence_indices, second_sentence_indices, label)
        self.line_id = line_id

    @overrides
    def as_testing_data(self, mode="word"):
        """
        Transforms the instance into a collection of NumPy
        arrays suitable for use as testing data in the model.

        Returns
        -------
        data_tuple: tuple
            The first element of this tuple has the NumPy array
            of the first sentence, and the second element has the
            NumPy array of the second sentence.

        mode: str, optional (default="word")
            String describing whether to return the word-level representations,
            character-level representations, or both. One of "word",
            "character", or "word+character"
        """
        if mode not in set(["word", "character", "word+character"]):
            raise ValueError("Input mode was {}, expected \"word\","
                             "\"character\", or \"word+character\"")
        if mode == "word" or mode == "word+character":
            first_sentence_word_array = np.asarray([word.word_index for word
                                                    in self.first_sentence_indices],
                                                   dtype="int32")
            second_sentence_word_array = np.asarray([word.word_index for word
                                                     in self.second_sentence_indices],
                                                    dtype="int32")
        if mode == "character" or mode == "word+character":
            first_sentence_char_matrix = np.asarray([word.char_indices for word
                                                     in self.first_sentence_indices],
                                                    dtype="int32")
            second_sentence_char_matrix = np.asarray([word.char_indices for word
                                                      in self.second_sentence_indices],
                                                     dtype="int32")
        if mode == "character":
            return ((first_sentence_char_matrix, second_sentence_char_matrix),
                    (),(np.asarray([self.line_id])))
        if mode == "word":
            return ((first_sentence_word_array, second_sentence_word_array),
                    (),(np.asarray([self.line_id])))
        if mode == "word+character":
            return ((first_sentence_word_array, first_sentence_char_matrix,
                     second_sentence_word_array, second_sentence_char_matrix),
                    (),(np.asarray([self.line_id])))
