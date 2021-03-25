import collections
import os
from typing import List, Dict, Set

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
    TRAIN_TAG = 'TRAIN'
    DOC_TAG = 'reuters'
    STATISTICS_OUT_DIR = 'statistics_out'

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
        # TODO add

    def normalize_text(self, text: str, remove_stop_words: bool = True, lemmatization: bool = True) -> List[str]:
        """Split text to sentences and tokens. Optional stop words removal and lemmatization. Returns list of tokens."""
        tokens = []
        for sentence in sent_tokenize(text):
            sentence_tokens = []
            for tok in self.tokenizer.tokenize(sentence):
                tok = tok.lower()
                if remove_stop_words and tok in self.stop_words:
                    continue
                if lemmatization:
                    tok = self.lemmatizer.lemmatize(tok)
                sentence_tokens.append(tok)

            tokens.extend(sentence_tokens)
        return tokens

    @staticmethod
    def _join_strings(strings: List[str], sep: str = ' ') -> str:
        return sep.join(strings)

    @staticmethod
    def write_data(docs: List[str], topics: List[str]):
        assert len(docs) == len(topics)
        pass

    @staticmethod
    def read_data():
        pass

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
    def _get_topics(document_tag: Tag) -> List[str]:
        topics = [topic_tag.text for topic_tag in document_tag.topics.contents]
        return topics

    @staticmethod
    def _get_is_train_split(document_tag: Tag) -> bool:
        is_train_split = document_tag.lewissplit == Preprocessor.TRAIN_TAG
        return is_train_split

    def preprocess_reuters_files(self, reuters_raw_path: str, min_doc_length: int = 1, save_stats: bool = False):
        assert os.path.exists(reuters_raw_path), LOGGER.warn(f'Reuters dataset not found at: {reuters_raw_path}!')
        train_docs, train_topics = [], []
        test_docs, test_topics = [], []

        chunk_paths = self._get_chunk_paths(reuters_raw_path)
        for chunk_file_path in chunk_paths:
            LOGGER.info(f"reading: {os.path.basename(chunk_file_path)}")
            with open(chunk_file_path, mode="rb") as chunk_reader:
                data = chunk_reader.read()
                data = data.decode("utf-8", "ignore")

            parsed_data = BeautifulSoup(data, features="lxml")
            document_tags = parsed_data.findAll(Preprocessor.DOC_TAG)

            for i, doc in enumerate(document_tags):
                # parse title, body and topics
                # title = self._get_title(doc)  # TODO test with titles
                body = self._get_body(doc)
                norm_body = self.normalize_text(body)
                # skip too short docs also docs without body
                if len(norm_body) < min_doc_length:
                    continue
                topics = self._get_topics(doc)
                is_train_split = self._get_is_train_split(doc)
                self._update_statistics(doc, topics, is_train_split)

                # append to train/ test
                norm_body = self._join_strings(norm_body)
                topics = self._join_strings(topics)
                if is_train_split:
                    train_docs.append(norm_body)
                    train_topics.append(topics)
                else:
                    test_docs.append(norm_body)
                    test_topics.append(topics)

                # print(title, ' - ', post_id, ' - ', is_train_split)
                # print(topics)
                # print(body, '\n\n')
                # if int(i) > 5:
                #     break
        self._save_statistics(save_stats)
        return (train_docs, train_topics), (test_docs, test_topics)

    def preprocess_text_file(self, text_file_path: str):
        pass


if __name__ == '__main__':
    reuters_raw_path_ = CONFIG['DATA']['reuters_raw_path']
    min_doc_length_ = CONFIG['DATA']['min_doc_length']
    save_stats_ = CONFIG['DATA']['save_stats']

    prep = Preprocessor()
    train, test = prep.preprocess_reuters_files(reuters_raw_path_, min_doc_length_, save_stats_)
