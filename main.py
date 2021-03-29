import argparse
import os
from typing import Optional, List

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as tf_metrics
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler

from config import CONFIG, parse_config_section
from constants import LOGGER, TOPICS_COUNTS
from data_preprocess import Preprocessor
from data_tf import TfDataHandler
from training_utils import TrainingUtils as Utils


tf.get_logger().setLevel('ERROR')


def get_model(model_name: str, model_path: Optional[str] = None) -> tf.keras.Sequential:
    vocabulary_size = CONFIG.getint('TfDataHandler', 'vocabulary_size')

    if model_name.lower().strip() == 'lstm':
        model = get_model_lstm(vocabulary_size)
    elif model_name.lower().strip() == 'term_freq':
        model = get_model_term_freq(vocabulary_size)
    else:
        raise ValueError(f'Unknown model_name: {model_name}! Supports: lstm or term_freq')

    if model_path is not None:
        model.load_weights(model_path).expect_partial()
        LOGGER.info(f' * Loaded {model_name} model from: {model_path}')
    model.summary()
    return model


def get_model_lstm(vocabulary_size: int, embedding_dim: int = 64, doc_max_len: int = 200, num_classes: int = 116) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=doc_max_len))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))

    # Classifier
    dropout_p = 0.25
    model.add(tf.keras.layers.Dense(512, activation='elu'))
    model.add(tf.keras.layers.Dropout(dropout_p))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='sigmoid'))
    return model


def get_model_term_freq(vocabulary_size: int, num_classes: int = 116) -> tf.keras.Sequential:
    """Text encoding vector of size vocabulary_size. Each value in vector is term frequency in the text."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=vocabulary_size))

    # Classifier
    dropout_p = 0.25
    model.add(tf.keras.layers.Dense(128, activation='elu'))
    model.add(tf.keras.layers.Dropout(dropout_p))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='sigmoid'))
    return model


def train(save_model_path: str = 'models/best_model', model_name: str = 'lstm'):
    # parse config
    train_prep_data_path = CONFIG.get('DATA', 'reuters_prep_train_path')
    test_prep_data_path = CONFIG.get('DATA', 'reuters_prep_test_path')
    tf_data_params = parse_config_section(CONFIG['TfDataHandler'])

    # get data
    train_docs, train_topics, _ = Preprocessor.read_data(train_prep_data_path)
    test_docs, test_topics, _ = Preprocessor.read_data(test_prep_data_path)
    tf_data = TfDataHandler(**tf_data_params)
    is_term_freq_data = True if model_name == 'term_freq' else False
    train_gen, test_gen = tf_data.get_tensorflow_data_generators(train_docs, train_topics, test_docs, test_topics, term_freq=is_term_freq_data)

    # train PARAMS
    learning_rate = 0.001
    epochs = 20

    # get model
    model = get_model(model_name=model_name)

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    metrics = ['acc',
               Utils.f1_score,
               tf_metrics.Precision(top_k=3, name='prec@3'),
               tf_metrics.Recall(top_k=3, name='rec@3')]
    model.compile(optimizer, loss='binary_crossentropy', metrics=metrics)

    # train
    save_best_callback = ModelCheckpoint(save_model_path, 'val_f1_score', save_best_only=True, save_weights_only=True, mode='max')
    lr_callback = LearningRateScheduler(lambda epoch_id, lr: lr if epoch_id < 15 else lr / 2)
    class_weight_dict = Utils.get_class_weights(TOPICS_COUNTS)
    history = model.fit(train_gen,
                        validation_data=test_gen,
                        epochs=epochs,
                        callbacks=[save_best_callback, lr_callback],
                        class_weight=class_weight_dict)

    # evaluate best model
    model.load_weights(save_model_path).expect_partial()
    model.evaluate(test_gen)
    text_paths = ['data/text_doc_grain_rice.txt', 'data/text_doc1.txt']
    label(text_paths, '', top_k=10, model=model)
    return history


def test(model_path: str, model_name: str = 'lstm'):
    # parse config
    test_prep_data_path = CONFIG.get('DATA', 'reuters_prep_test_path')
    tf_data_params = parse_config_section(CONFIG['TfDataHandler'])

    # get data
    test_docs, test_topics, _ = Preprocessor.read_data(test_prep_data_path)
    tf_data = TfDataHandler(**tf_data_params)
    _, test_gen = tf_data.get_tensorflow_data_generators([''], [''], test_docs, test_topics)

    # get model
    model = get_model(model_name, model_path)

    # evaluate
    metrics = ['acc',
               Utils.f1_score,
               tf_metrics.Precision(top_k=3, name='prec@3'),
               tf_metrics.Recall(top_k=3, name='rec@3')]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    model.evaluate(test_gen)


def label(text_paths: List[str], model_path: str, model_name: str = 'lstm', top_k: int = 5, model=None):
    prep = Preprocessor()

    tf_data_params = parse_config_section(CONFIG['TfDataHandler'])
    tf_data = TfDataHandler(**tf_data_params)
    tf_data.setup_tokenizers()

    if model is None:
        model = get_model(model_name, model_path)
        model.trainable = False

    for text_path in text_paths:
        text = prep.preprocess_text_file(text_path)
        is_term_freq_data = True if model_name == 'term_freq' else False
        tf_ready_data, _ = tf_data.prepare_data_for_tensorflow([text], [''], term_freq=is_term_freq_data)
        tf_ready_data = tf_ready_data.reshape((1, -1))
        predictions = model(tf_ready_data)
        predictions = predictions.numpy().flatten()
        top_k_ids = np.argsort(predictions)[::-1][:top_k]
        top_k_labels = [tf_data.label_tokenizer.index_word[label_id] for label_id in top_k_ids]
        top_k_scores = np.round(predictions[top_k_ids], 2)

        print(f'\nTEXT from: {text_path}')
        print(f'** TOP {top_k} LABELS ** \n{top_k_labels}\n{top_k_scores}')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, help='train test or label')
    parser.add_argument('--model_path', required=True, help='Path to model weights.')
    parser.add_argument('--model_name', help='lstm or term_freq', default='lstm')
    parser.add_argument('--input_text_file')
    return parser.parse_args()


if __name__ == '__main__':
    Utils.setup_tf()
    args = get_args()
    running_mode = args.mode.lower().strip()
    assert running_mode in ['train', 'test', 'label'], print(f'Not supported running mode! {running_mode}')

    LOGGER.info(f'[MAIN] - {running_mode.upper()}')
    if running_mode == 'train':
        train(save_model_path=args.model_path,  model_name=args.model_name)

    elif running_mode == 'test':
        test(model_path=args.model_path, model_name=args.model_name)

    elif running_mode == 'label':
        assert os.path.exists(args.input_text_file), print(f'Input path NOT exists!')
        input_text_file = [args.input_text_file]
        label(input_text_file, model_path=args.model_path, model_name=args.model_name)
