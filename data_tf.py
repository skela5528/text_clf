import os
import pickle
from typing import List, Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from constants import LOGGER


class TfDataHandler:
    """Convert normalized textual data to tensorflow format: text to sequence of term_ids, topics to one_hot vector."""

    def __init__(self, vocabulary_size: int, doc_max_len: int, batch_size: int,
                 data_tokenizer_path: str, labels_tokenizer_path: str):
        self.vocabulary_size = vocabulary_size
        self.doc_max_len = doc_max_len
        self.batch_size = batch_size
        self.oov_token = '<OOV>'
        self.data_tokenizer_path = data_tokenizer_path
        self.labels_tokenizer_path = labels_tokenizer_path

        self.data_tokenizer = None   # type: Optional[Tokenizer]
        self.label_tokenizer = None  # type: Optional[Tokenizer]

    def setup_tokenizers(self, docs: Optional[List[str]] = None, topics: Optional[List[str]] = None, overwrite: bool = False) -> bool:
        if os.path.exists(self.data_tokenizer_path) and not overwrite:
            self.data_tokenizer = self.read_tokenizer(self.data_tokenizer_path)
        else:
            self.data_tokenizer = Tokenizer(num_words=self.vocabulary_size, filters='', lower=False, oov_token=self.oov_token)
            self.data_tokenizer.fit_on_texts(docs)
            self.write_tokenizer(self.data_tokenizer, self.data_tokenizer_path)

        if os.path.exists(self.labels_tokenizer_path) and not overwrite:
            self.label_tokenizer = self.read_tokenizer(self.labels_tokenizer_path)
        else:
            self.label_tokenizer = Tokenizer(filters='', lower=False)
            self.label_tokenizer.fit_on_texts(topics)
            self.write_tokenizer(self.label_tokenizer, self.labels_tokenizer_path)
        is_setup_succeed = self.data_tokenizer is not None and self.label_tokenizer is not None
        LOGGER.info(f'[data-tf] data_tokenizer num_words: {self.data_tokenizer.num_words}')
        return is_setup_succeed

    def prepare_data_for_tensorflow(self, docs: List[str], topics: List[str],
                                    term_freq: bool = False, exclude_no_topic: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        LOGGER.info(f'[data-tf] prepare {len(docs)} docs | term_freq:{term_freq} ')
        if not term_freq:
            doc_sequences = self.data_tokenizer.texts_to_sequences(docs)
            doc_sequences = pad_sequences(doc_sequences, maxlen=self.doc_max_len, padding='post', truncating='post')
        else:
            doc_sequences = self.data_tokenizer.texts_to_matrix(docs, mode='freq')
        labels_sequences = self.label_tokenizer.texts_to_matrix(topics)

        if exclude_no_topic:
            include_ids = labels_sequences[:,  1] == 0
            doc_sequences = doc_sequences[include_ids]
            labels_sequences = labels_sequences[include_ids]
        return doc_sequences, labels_sequences

    @staticmethod
    def write_tokenizer(tokenizer: Tokenizer, out_path: str):
        LOGGER.info(f'[data-tf] write tokenizer to: {out_path}')
        with open(out_path, mode='wb') as stream:
            pickle.dump(tokenizer, stream)

    @staticmethod
    def read_tokenizer(tokenizer_path: str) -> Tokenizer:
        assert os.path.exists(tokenizer_path)
        LOGGER.info(f'[data-tf] read tokenizer from: {tokenizer_path}')
        with open(tokenizer_path, mode='rb') as stream:
            tokenizer = pickle.load(stream)
        return tokenizer

    def get_tensorflow_data_generators(self, train_docs: List[str], train_topics: List[str],
                                       test_docs: List[str], test_topics: List[str],
                                       term_freq: bool = False, exclude_no_topic: bool = False) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        self.setup_tokenizers(train_docs, train_topics)
        train_docs, train_topics = self.prepare_data_for_tensorflow(train_docs, train_topics, term_freq, exclude_no_topic)
        test_docs, test_topics = self.prepare_data_for_tensorflow(test_docs, test_topics, term_freq, exclude_no_topic)
        LOGGER.info(f'[data-tf] train size: {len(train_docs)}')
        LOGGER.info(f'[data-tf] test size: {len(test_docs)}')

        train_data = train_docs.tolist(), train_topics
        test_data = test_docs.tolist(), test_topics

        train_data_generator = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(self.batch_size)
        test_data_generator = tf.data.Dataset.from_tensor_slices(test_data).batch(self.batch_size)
        return train_data_generator, test_data_generator
