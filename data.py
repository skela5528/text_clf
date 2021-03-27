import collections
import os
import pickle
from typing import List, Dict, Set, Tuple

import nltk
from bs4 import BeautifulSoup
from bs4.element import Tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer, sent_tokenize

from config import CONFIG
from constants import LOGGER

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


class Preprocessor:
    BODY_ID = 4
    TEXT_TAG = 'text'
    TITLE_TAG = 'title'
    TRAIN_TAG = 'lewissplit'
    TRAIN_SPLIT_STRING = 'TRAIN'
    DOC_TAG = 'reuters'
    STATISTICS_OUT_DIR = '../statistics_out'
    NO_TOPIC = 'NO_TOPIC'

    def __init__(self):
        self.stop_words = set(stopwords.words('english'))  # type: Set[str]
        self.stop_words.add('reuter')
        self.tokenizer = RegexpTokenizer('[\'a-zA-Z]+')
        self.lemmatizer = WordNetLemmatizer()
        self.worlds_counter = collections.defaultdict(int)      # type: Dict[str, int]
        self.topics_counter = collections.defaultdict(int)      # type: Dict[str, int]
        self.doc_lengths = []

    def _update_statistics(self, doc: List[str], topics: List[str], is_train: bool):
        def add_to_count(count: dict, values: list):
            for val in values:
                count[val] += 1
        if is_train:
            add_to_count(self.worlds_counter, doc)
            add_to_count(self.topics_counter, topics)
            self.doc_lengths.append(len(doc))

    def _save_statistics(self, save_stats: bool):
        if not save_stats:
            return
        out_path = os.path.join(self.STATISTICS_OUT_DIR, 'vocab.pckl')
        os.makedirs(self.STATISTICS_OUT_DIR, exist_ok=True)
        with open(out_path, 'wb') as stream:
            pickle.dump(self.worlds_counter, stream)
        # TODO add other stats ?

    def normalize_text(self, text: str, remove_stop_words: bool = True, lemmatization: bool = True) -> List[str]:
        """Split text to sentences and tokens. Optional stop words removal and lemmatization. Returns list of tokens."""
        tokens = []
        for sentence in sent_tokenize(text):
            sentence_tokens = []
            for tok in self.tokenizer.tokenize(sentence):
                # lower
                tok = tok.lower()
                # stop words
                if remove_stop_words and tok in self.stop_words:
                    continue
                # lemmatization
                if lemmatization:
                    tok = self.lemmatizer.lemmatize(tok)
                sentence_tokens.append(tok)

            tokens.extend(sentence_tokens)
        return tokens

    @staticmethod
    def _join_strings(strings: List[str], sep: str = ' ') -> str:
        return sep.join(strings)

    @staticmethod
    def write_data(docs: List[str], topics: List[str], ids: List[str], out_path: str, sep: str = ','):
        assert len(docs) == len(topics) == len(ids)
        LOGGER.info(f'[data] writing normalized data to {os.path.abspath(out_path)}')
        with open(out_path, mode='a') as stream:
            for doc, doc_topics, doc_newid in zip(docs, topics, ids):
                line = f'{doc}{sep}{doc_topics}{sep}{doc_newid}\n'
                stream.write(line)

    @staticmethod
    def read_data(data_path: str, sep: str = ',') -> Tuple[List[str], List[str], List[str]]:
        assert os.path.exists(data_path)
        LOGGER.info(f'[data] writing normalized data to {os.path.abspath(data_path)}')
        docs, topics, ids = [], [], []
        with open(data_path) as stream:
            lines = stream.readlines()
            for line in lines:
                doc, doc_topics, doc_newid = line.strip().split(sep)
                docs.append(doc)
                topics.append(doc_topics)
                ids.append(doc_newid)
        return docs, topics, ids

    @staticmethod
    def _get_chunk_paths(reuters_raw_path: str, chunk_ext: str = '.sgm') -> List[str]:
        file_names = os.listdir(reuters_raw_path)
        file_names.sort()
        chunk_names = [name for name in file_names if name.endswith(chunk_ext)]
        chunk_paths = [os.path.join(reuters_raw_path, name) for name in chunk_names]
        return chunk_paths

    @staticmethod
    def _get_title(document_tag: Tag) -> str:
        try:
            title = document_tag(Preprocessor.TEXT_TAG)[0](Preprocessor.TITLE_TAG)[0].text
        except IndexError:
            title = ""
        return title

    @staticmethod
    def _get_body(document_tag: Tag) -> str:
        doc_contents = document_tag(Preprocessor.TEXT_TAG)[0].contents  # type: list
        try:
            body = str(doc_contents[Preprocessor.BODY_ID])
        except IndexError:
            body = ""
        return body

    @staticmethod
    def _get_topics(document_tag: Tag, no_topic: str) -> List[str]:
        topics = [topic_tag.text for topic_tag in document_tag.topics.contents]
        topics = topics if len(topics) else [no_topic]
        return topics

    @staticmethod
    def _get_is_train_split(document_tag: Tag) -> bool:
        is_train_split = document_tag[Preprocessor.TRAIN_TAG] == Preprocessor.TRAIN_SPLIT_STRING
        return is_train_split

    def preprocess_reuters_files(self, reuters_raw_path: str, min_doc_length: int = 1, save_stats: bool = False):
        assert os.path.exists(reuters_raw_path), LOGGER.warn(f'Reuters dataset not found at: {reuters_raw_path}!')
        train_ids, train_docs, train_topics = [], [], []
        test_ids, test_docs, test_topics = [], [], []

        chunk_paths = self._get_chunk_paths(reuters_raw_path)
        for chunk_i, chunk_file_path in enumerate(chunk_paths):
            LOGGER.info(f'[data] reading: {os.path.basename(chunk_file_path)}')
            with open(chunk_file_path, mode='rb') as chunk_reader:
                data = chunk_reader.read()
                data = data.decode('utf-8', 'ignore')

            parsed_data = BeautifulSoup(data, features='lxml')
            document_tags = parsed_data.findAll(Preprocessor.DOC_TAG)

            for doc_i, doc in enumerate(document_tags):
                # parse title, body and topics
                # title = self._get_title(doc)  # TODO test with titles
                body = self._get_body(doc)
                norm_body = self.normalize_text(body)
                # skip too short docs also docs without body
                if len(norm_body) < min_doc_length:
                    continue
                topics = self._get_topics(doc, self.NO_TOPIC)
                is_train_split = self._get_is_train_split(doc)
                self._update_statistics(norm_body, topics, is_train_split)

                # append to train/ test
                norm_body = self._join_strings(norm_body)
                topics = self._join_strings(topics)
                if is_train_split:
                    train_docs.append(norm_body)
                    train_topics.append(topics)
                    train_ids.append(doc['newid'])
                else:
                    test_docs.append(norm_body)
                    test_topics.append(topics)
                    test_ids.append(doc['newid'])

        self._save_statistics(save_stats)
        return (train_docs, train_topics, train_ids), (test_docs, test_topics, test_ids)

    def preprocess_text_file(self, text_file_path: str):
        pass


def prepare_data():
    # parse config
    reuters_raw_path = CONFIG.get('DATA', 'reuters_raw_path')
    min_doc_length = CONFIG.getint('DATA', 'min_doc_length')
    save_stats = CONFIG.get('DATA', 'save_stats')
    train_prep_data_path = CONFIG.get('DATA', 'reuters_prep_train_path')
    test_prep_data_path = CONFIG.get('DATA', 'reuters_prep_test_path')

    # normalize data
    prep = Preprocessor()
    train, test = prep.preprocess_reuters_files(reuters_raw_path, min_doc_length, save_stats)

    # save preprocessed data
    prep.write_data(*train, train_prep_data_path)
    prep.write_data(*test, test_prep_data_path)


if __name__ == '__main__':
    prepare_data()
