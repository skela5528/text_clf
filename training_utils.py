from collections import Iterable
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.python.keras.backend

from data_tf import TfDataHandler


class TrainingUtils:
    @staticmethod
    def save_model(model: tf.keras.Model):
        time_now = datetime.now().strftime("%H_%M_%S")
        save_path = f'model_{time_now}.h5'
        model.save(save_path, include_optimizer=False, save_format='h5')

    @staticmethod
    def load_model(model_path: str):
        model = tf.keras.models.load_model(model_path)  # type: tf.keras.Model
        return model

    @staticmethod
    def check_data(train_data_generator: tf.data.Dataset, tf_data: TfDataHandler):
        """Sanity check for data after all conversion to matrix form."""

        text = ''
        labels = ''
        in_batch_id = 1
        for i, (batch_docs, batch_labels) in enumerate(train_data_generator):
            if i > 0:
                break
            print(batch_docs.shape, '\n', batch_docs[in_batch_id])
            print(batch_labels.shape, '\n', batch_labels[in_batch_id])

            text = []
            for tok_id in batch_docs[in_batch_id].numpy():
                if tok_id == 0:
                    continue
                tok = tf_data.data_tokenizer.index_word[tok_id]
                text.append(tok)

            labels_one_hot = batch_labels[in_batch_id].numpy()
            labels_ids = np.where(labels_one_hot == 1)[0]
            labels = [tf_data.label_tokenizer.index_word[lid] for lid in labels_ids]
        print(text)
        print(labels)

    @staticmethod
    def setup_tf():
        import sys

        # init
        tf.python.keras.backend.clear_session()
        tf.random.set_seed(51)
        np.random.seed(51)

        # print info
        print(f'* check environment *')
        print(f' - python: {sys.version}')
        print(f" - tf version: {tf.__version__}")
        print(f" - tf is_built_with_cuda: {tf.test.is_built_with_cuda()}")
        tf.config.list_physical_devices('GPU')
        print()

    @staticmethod
    def f1_score(y_true: Iterable, y_pred: Iterable):
        """Compute the micro f(b) score with b=1.
        Taken from here: https://github.com/AlexGidiotis/Document-Classifier-LSTM/blob/master/hatt_classifier.py"""

        y_true = tf.cast(y_true, 'float32')
        y_pred = tf.cast(tf.round(y_pred), 'float32')  # implicit 0.5 threshold via tf.round
        y_correct = y_true * y_pred

        sum_true = tf.reduce_sum(y_true, axis=1)
        sum_pred = tf.reduce_sum(y_pred, axis=1)
        sum_correct = tf.reduce_sum(y_correct, axis=1)

        precision = sum_correct / sum_pred
        recall = sum_correct / sum_true
        f_score = 2 * precision * recall / (precision + recall)
        f_score = tf.where(tf.math.is_nan(f_score), tf.zeros_like(f_score), f_score)

        return tf.reduce_mean(f_score)

    @staticmethod
    def get_class_weights(counts: np.ndarray) -> dict:
        """Produce lower weight for frequent classes."""
        weights = np.log(.1 * sum(counts) / counts)
        weights[weights < 1] = 1
        class_weight_dict = dict(zip(range(1, len(counts) + 1), weights))
        class_weight_dict[0] = .01
        return class_weight_dict
