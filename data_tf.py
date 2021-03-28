import os
from typing import List, Tuple, Optional
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from constants import LOGGER


class TfDataHandler:
    """Convert normalized textual data to tensorflow format: text to sequence of term_ids, topics to one_hot vector."""

    def __init__(self, vocabulary_size: int, doc_max_len: int, batch_size: int):
        self.vocabulary_size = vocabulary_size
        self.doc_max_len = doc_max_len
        self.batch_size = batch_size
        self.oov_token = '<OOV>'

        self.data_tokenizer = None   # type: Optional[Tokenizer]
        self.label_tokenizer = None  # type: Optional[Tokenizer]

    def _setup_tokenizers(self, docs: List[str], topics: List[str]):
        self.data_tokenizer = Tokenizer(num_words=self.vocabulary_size, filters='', lower=False, oov_token=self.oov_token)
        self.data_tokenizer.fit_on_texts(docs)

        self.label_tokenizer = Tokenizer(filters='', lower=False)
        self.label_tokenizer.fit_on_texts(topics)

    def _prepare_data_for_tensorflow(self, docs: List[str], topics: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        LOGGER.info(f'[data-tf] prepare {len(docs)} docs')
        doc_sequences = self.data_tokenizer.texts_to_sequences(docs)
        doc_sequences = pad_sequences(doc_sequences, maxlen=self.doc_max_len, padding='post', truncating='post')
        labels_sequences = self.label_tokenizer.texts_to_matrix(topics)
        return doc_sequences, labels_sequences

    def write_tokenizers(self, out_dir: str):
        # TODO
        pass

    def read_tokenizer(self, tokenizer_path: str) -> Tokenizer:
        # TODO
        pass

    def get_tensorflow_data_generators(self, train_docs: List[str], train_topics: List[str],
                                       test_docs: List[str], test_topics: List[str]) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        self._setup_tokenizers(train_docs, train_topics)
        train_docs, train_topics = self._prepare_data_for_tensorflow(train_docs, train_topics)
        test_docs, test_topics = self._prepare_data_for_tensorflow(test_docs, test_topics)

        train_data = train_docs.tolist(), train_topics
        test_data = test_docs.tolist(), test_topics

        train_data_generator = tf.data.Dataset.from_tensor_slices(train_data).shuffle(10000).batch(self.batch_size)
        test_data_generator = tf.data.Dataset.from_tensor_slices(test_data).batch(self.batch_size)

        return train_data_generator, test_data_generator
