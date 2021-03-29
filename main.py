import argparse
import os
from typing import Optional, List

import numpy as np
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
import tensorflow as tf
import tensorflow.keras.metrics as tf_metrics

from data_preprocess import Preprocessor
from data_tf import TfDataHandler
from config import CONFIG, parse_config_section
from constants import LOGGER, TOPICS_COUNTS
from training_utils import TrainingUtils as Utils


def get_model(model_name: str, model_path: Optional[str] = None) -> tf.keras.Sequential:
    if model_name.lower().strip() == 'lstm':
        model = get_model_lstm()
    elif model_name.lower().strip() == 'term_freq':
        model = get_model_term_freq()
    else:
        raise ValueError(f'Unknown model_name: {model_name}! Supports: lstm or term_freq')

    if model_path is not None:
        model.load_weights(model_path).expect_partial()
        LOGGER.info(f' * Loaded {model_name} model from: {model_path}')
    model.summary()
    return model


def get_model_lstm(vocabulary_size: int = 5000, embedding_dim: int = 64, doc_max_len: int = 200, num_classes: int = 116) -> tf.keras.Sequential:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(input_dim=vocabulary_size, output_dim=embedding_dim, input_length=doc_max_len))

    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)))

    # Classifier
    dropout_p = 0.25
    model.add(tf.keras.layers.Dense(512, activation='elu'))
    model.add(tf.keras.layers.Dropout(dropout_p))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))  # TODO revert sigmoid
    return model


def get_model_term_freq(vocabulary_size: int = 5000, num_classes: int = 116) -> tf.keras.Sequential:
    """Text encoding vector of size vocabulary_size. Each value in vector is term frequency in the text."""
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.InputLayer(input_shape=vocabulary_size))

    # Classifier
    dropout_p = 0.25
    model.add(tf.keras.layers.Dense(128, activation='elu'))
    model.add(tf.keras.layers.Dropout(dropout_p))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='sigmoid'))
    return model


def test_model(model: Optional[tf.keras.Sequential], tf_data: TfDataHandler):
    if model is None:
        model = get_model_lstm()
        model.load_weights("models/best_model").expect_partial()

    model.trainable = False
    data_path = '/home/cortica/Documents/my/git_personal/data/text_doc.txt'
    prep = Preprocessor()
    # text = prep.preprocess_text_file(data_path)
    text = "u agriculture secretary richard lyng warned japan failure remove longstanding import quota japanese beef might spark protectionist response united state given protectionist mood congress country leader japan would certainly concerned failure remove beef quota might serious lyng told group u cattleman lyng said trade representative clayton yeutter visit japan later month demand total elimination beef import quota april current dispute japan semiconductor may strengthen u stance farm trade negotiation lyng said japan want trade war u lyng dismissed recent statement tokyo japan might retaliate u product result semiconductor dispute japan going pick fight u lyng said adding huge bilateral trade surplus japan lose trade war united state lyng told u cattleman quota japanese beef import allow consumer adequate choice food purchase said addition beef u press eliminiation import barrier japan's citrus rice well lyng noted japan largest buyer u farm product principally grain soybean"
    print(text)

    tf_ready_data, _ = tf_data.prepare_data_for_tensorflow([text], [''])
    tf_ready_data = tf_ready_data.reshape((1, -1))
    predictions = model(tf_ready_data)

    # predictions
    predictions = predictions.numpy().flatten()
    top_k = 10
    top_k_ids = np.argsort(predictions)[::-1][:top_k]
    top_k_labels = [tf_data.label_tokenizer.index_word[label_id] for label_id in top_k_ids]

    print(top_k_labels)
    print(predictions[top_k_ids])


def train(save_model_path: str = 'models/best_model', model_name: str = 'lstm'):
    # parse config
    train_prep_data_path = CONFIG.get('DATA', 'reuters_prep_train_path')
    test_prep_data_path = CONFIG.get('DATA', 'reuters_prep_test_path')
    tf_data_params = parse_config_section(CONFIG['TfDataHandler'])

    # get data
    train_docs, train_topics, _ = Preprocessor.read_data(train_prep_data_path)
    test_docs, test_topics, _ = Preprocessor.read_data(test_prep_data_path)
    tf_data = TfDataHandler(**tf_data_params)
    train_gen, test_gen = tf_data.get_tensorflow_data_generators(train_docs, train_topics, test_docs, test_topics)

    # Utils.check_data(test_gen, tf_data)

    # train PARAMS
    LR = 0.001  # learning rate
    EP = 20     # number of epochs

    # get model
    model = get_model(model_name=model_name)

    # compile model
    optimizer = tf.keras.optimizers.Adam(learning_rate=LR)
    metrics = ['acc', Utils.f1_score, tf_metrics.Precision(top_k=3, name='prec@3'), tf_metrics.Recall(top_k=3, name='rec@3')]
    model.compile(optimizer, loss='binary_crossentropy', metrics=metrics)

    save_best_callback = tf.keras.callbacks.ModelCheckpoint(filepath=save_model_path,
                                                            save_weights_only=True,
                                                            monitor='val_f1_score',
                                                            save_best_only=True)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lambda ep, lr: lr if ep < 15 else lr / 2)

    # train
    class_weight_dict = Utils.get_class_weights(TOPICS_COUNTS)
    history = model.fit(train_gen,
                        validation_data=test_gen,
                        epochs=EP,
                        callbacks=[save_best_callback, lr_callback],
                        class_weight=class_weight_dict)


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
    metrics = ['acc', Utils.f1_score, tf_metrics.Precision(top_k=3, name='prec@3'), tf_metrics.Recall(top_k=3, name='rec@3')]
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
    model.evaluate(test_gen)


def label(text_paths: List[str], model_path: str, model_name: str = 'lstm', top_k: int = 5):
    prep = Preprocessor()

    tf_data_params = parse_config_section(CONFIG['TfDataHandler'])
    tf_data = TfDataHandler(**tf_data_params)
    tf_data.setup_tokenizers()

    model = get_model(model_name, model_path)
    model.trainable = False

    for text_path in text_paths:
        text = prep.preprocess_text_file(text_path)

        tf_ready_data, _ = tf_data.prepare_data_for_tensorflow([text], [''])
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

    if running_mode == 'train':
        train()

    elif running_mode == 'test':
        test(model_path=args.model_path)

    elif running_mode == 'label':
        assert os.path.exists(args.input_text_file), print(f'Input path NOT exists!')
        input_text_file = [args.input_text_file]
        label(input_text_file, model_path=args.model_path, model_name=args.model_name)
