import csv
import json
import os
import random
from collections import Counter, defaultdict
import math
import operator

import logging
import nltk
import numpy as np
from tqdm import tqdm


class Data(object):
    def __init__(self, config, train_data=None):
        self.config = config
        self.batch_size = config.batch_size
        if train_data is not None:
            assert isinstance(train_data, self.__class__)
        self.train_data = train_data
        self.shapes = []
        self.dtypes = []
        self.names = []
        self._word2vec_dict = None

    def __iter__(self):
        """
        Get next element in feed dict format
        :return:
        """
        raise NotImplementedError()


class SquadData(Data):
    def __init__(self, config, train_data=None):
        super(SquadData, self).__init__(config, train_data=train_data)
        self.data_type = config.data_type
        self.names = ['x', 'q', 'y1', 'y2', 'x_len', 'q_len', 'idxs']
        if config.serve:
            self._common = dict()
            self._data = defaultdict(list)
            self._shared = defaultdict(list)
            self.squad = {'data': []}
            _ = self.word2vec_dict  # for pre-load purpose
        else:
            self.load(update=True)
        self.shapes = {'x': [config.max_context_size], 'q': [config.max_ques_size], 'y1': [], 'y2': [],
                       'x_len': [], 'q_len': [], 'idxs': []}
        self.dtypes = {'x': 'int32', 'q': 'int32', 'y1': 'int32', 'y2': 'int32', 'x_len': 'int32', 'q_len': 'int32', 'idxs': 'int64'}

    def load(self, update=False):
        config = self.config
        with open(config.squad_path, 'r') as fp:
            self.squad = json.load(fp)
        self._prepro(self.squad)
        self._common = self._get_common(self.squad)
        metadata = self._get_metadata(self.squad)
        self._data, self._shared = self._get_data(self.squad, metadata)
        # config.vocab_size = len(self._metadata['word2idx'])  # self._common['vocab_size']
        config.emb_mat = np.array(metadata['emb_mat'], dtype='float32')
        if update:
            config.max_context_size = max(len(xi) for xi in self._shared['x'])  # self._metadata['max_context_size']
            config.max_ques_size = max(len(qi) for qi in self._data['q'])  # self._metadata['max_ques_size']

    @property
    def num_examples(self):
        return len(self._data['q'])

    @property
    def num_batches(self):
        return int(math.ceil(self.num_examples/self.config.batch_size))

    @property
    def last_batch_size(self):
        return self.num_examples % self.config.batch_size

    def __iter__(self):
        for i in range(len(self._data['q'])):
            yield self.get(i)

    def get(self, i):
        each = {}
        for key in self._data:
            val = self._data[key][i]
            if key.startswith("*"):
                key = key[1:]
                val = self._shared[key][val]
            if key == 'x':
                val = idxs2np(val, self.config.max_context_size)
            elif key == 'q':
                val = idxs2np(val, self.config.max_ques_size)
            each[key] = val
        each['y1'] = self._data['y1s'][i][0]  # random.choice(self._data['y1s'][i])
        each['y2'] = self._data['y2s'][i][0]  # random.choice(self._data['y2s'][i])
        return each

    def _prepro(self, squad):
        if not self.config.fresh and os.path.exists(self.config.common_path) and os.path.exists(self.config.metadata_path) and os.path.exists(self.config.data_path):
            return

        for article in tqdm(squad['data'], desc="{} prepro".format(self.config.data_type)):
            for para in article['paragraphs']:
                context = process_text(para['context'])
                context_words = word_tokenize(context)
                para['processed_context'] = context
                para['context_words'] = context_words
                for qa in para['qas']:
                    ques = process_text(qa['question'])
                    ques_words = word_tokenize(ques)
                    qa['processed_question'] = ques
                    qa['question_words'] = ques_words
            if self.config.draft:
                break

    def _get_common(self, squad):
        if self.train_data is not None:
            common = self.train_data._common
            return common
        if os.path.exists(self.config.common_path):
            with open(self.config.common_path, 'r') as fp:
                print("Loading common info at {}".format(self.config.common_path))
                common = json.load(fp)
                return common
        assert self.config.train, "Need common file at {} for validation or test.".format(self.config.common_path)

        word_counter = Counter()

        for article in tqdm(squad['data'], desc="{} get_common".format(self.config.data_type)):
            for para in article['paragraphs']:
                context_words = para['context_words']
                for word in context_words:
                    word_counter[word] += len(para['qas'])
                for qa in para['qas']:
                    ques_words = qa['question_words']
                    for word in ques_words:
                        word_counter[word] += 1
            if self.config.draft:
                break

        vocab_words = ['<PAD>', '<UNK>'] + list(set(word for word, count in word_counter.items() if count >= self.config.word_count_th))
        word2idx_dict = {word: idx for idx, word in enumerate(vocab_words)}

        common = {'word2idx': word2idx_dict, 'vocab_size': len(word2idx_dict)}
        print("Dumping common at {}".format(self.config.common_path))
        with open(self.config.common_path, 'w') as fp:
            json.dump(common, fp)

        print("Dumping emb_metadata at {}".format(self.config.emb_metadata_path))
        with open(self.config.emb_metadata_path, 'w') as fp:
            writer = csv.writer(fp, delimiter='\t')
            writer.writerows([[word] for word in vocab_words])
        return common

    def _get_metadata(self, squad):
        if not self.config.fresh and os.path.exists(self.config.metadata_path):
            with open(self.config.metadata_path, 'r') as fp:
                print("Loading metadata info at {}".format(self.config.metadata_path))
                metadata = json.load(fp)
                return metadata

        max_context_size = 0
        max_ques_size = 0

        words = set()
        for article in tqdm(squad['data'], desc="{} get_metadata".format(self.config.data_type)):
            for para in article['paragraphs']:
                context_words = para['context_words']
                max_context_size = max(max_context_size, len(context_words))
                words |= set(context_words)
                for qa in para['qas']:
                    ques_words = qa['question_words']
                    max_ques_size = max(max_ques_size, len(ques_words))
                    words |= set(ques_words)
            if self.config.draft:
                break

        word2vec_dict = {word.lower(): self.word2vec_dict[word.lower()] for word in words if word in self.word2vec_dict}
        print("{}/{} words found in GloVe.".format(len(word2vec_dict), len(words)))
        vocab = ['<PAD>', '<UNK>'] + list(word2vec_dict)
        vec_size = len(next(iter(self.word2vec_dict.values())))
        word2vec_dict[vocab[0]] = [0.0] * vec_size
        word2vec_dict[vocab[1]] = [0.0] * vec_size
        idx2word_dict = {idx: word for idx, word in enumerate(vocab)}
        word2idx_dict = {idx: word for word, idx in idx2word_dict.items()}
        emb_mat = [word2vec_dict[idx2word_dict[idx]] for idx in range(len(vocab))]

        metadata = {'emb_mat': emb_mat, 'word2idx': word2idx_dict}
        with open(self.config.metadata_path, 'w') as fp:
            print("Dumping metadata at {}".format(self.config.metadata_path))
            json.dump(metadata, fp)
        return metadata

    def _get_data(self, squad, metadata):
        if not self.config.fresh and os.path.exists(self.config.data_path):
            with open(self.config.data_path, 'r') as fp:
                print("Loading data at {}".format(self.config.data_path))
                data, shared = json.load(fp)
                return data, shared

        # Switch this back to commented one for regular word indexing (not using glove)
        # TODO : enable switching between two methods
        word2idx_dict = metadata['word2idx']  # self._common['word2idx']

        x, rx, q, y1, y2 = [], [], [], [], []
        x_len, q_len = [], []
        sx, sx_len = [], []
        context_list, ques_list, ans_list, ids, idxs = [], [], [], [], []
        context_words_list, ques_words_list = [], []
        for article in tqdm(squad['data'], desc="{} get_data".format(self.config.data_type)):
            for para in article['paragraphs']:
                context = para['processed_context']
                context_words = para['context_words']
                xj = [word2idx(word2idx_dict, word) for word in context_words]
                xj_len = len(context_words)
                for qa in para['qas']:
                    rxi = len(x)
                    ques = qa['processed_question']
                    id_ = qa['id']
                    idx = len(q)
                    ques_words = qa['question_words']
                    qi = [word2idx(word2idx_dict, word) for word in ques_words]
                    qi_len = len(ques_words)

                    ans_text, yi1, yi2 = [], [], []
                    for ans in qa['answers']:
                        each_ans_start = ans['answer_start']
                        each_ans_stop = each_ans_start + len(ans['text'])
                        each_ans_text = context[each_ans_start:each_ans_stop]
                        each_yi1, each_yi2 = get_word_span(context, context_words, each_ans_start, each_ans_stop)
                        ans_text.append(each_ans_text)
                        yi1.append(each_yi1)
                        yi2.append(each_yi2)

                        phrase = get_phrase(context, context_words, (each_yi1, each_yi2))
                        if phrase != each_ans_text:
                            # print("'{}' != '{}'".format(phrase, each_ans_text))
                            logging.log(logging.DEBUG, "'{}' != '{}'".format(phrase, each_ans_text))

                    q.append(qi)
                    q_len.append(qi_len)
                    rx.append(rxi)
                    ques_list.append(ques)
                    ids.append(id_)
                    idxs.append(idx)
                    ans_list.append(ans_text)
                    y1.append(yi1)
                    y2.append(yi2)
                    ques_words_list.append(ques_words)
                x.append(xj)
                x_len.append(xj_len)
                context_list.append(context)
                context_words_list.append(context_words)
            if self.config.draft:
                break

        shared = {'x': x, 'x_len': x_len, 'context': context_list, 'context_words': context_words_list}
        data = {'*x': rx, 'q': q, 'y1s': y1, 'y2s': y2, '*x_len': rx, 'q_len': q_len,
                '*context': rx, '*context_words': rx, 'ques': ques_list, 'ans': ans_list, 'ques_words': ques_words_list,
                'ids': ids, 'idxs': idxs}

        print("Dumping data at {}".format(self.config.data_path))
        with open(self.config.data_path, 'w') as fp:
            json.dump([data, shared], fp)

        return data, shared

    @property
    def word2vec_dict(self):
        if self._word2vec_dict is None:
            if self.train_data is not None:
                self._word2vec_dict = self.train_data.word2vec_dict
            else:
                self._word2vec_dict = get_word2vec(self.config.glove_path, draft=self.config.draft)
        return self._word2vec_dict


def process_text(text):
    return text.replace("``", '" ').replace("''", '" ')


def word_tokenize(text):
    return [token.replace("``", '"').replace("''", '"') for token in nltk.word_tokenize(text)]


def word2idx(word2idx_dict, word):
    if word.lower() not in word2idx_dict:
        return 1
    return word2idx_dict[word.lower()]


def pad(l, size, pad_val=0):
    if len(l) > size:
        raise TooLongError()
    width = size - len(l)
    out = np.lib.pad(l, (0, width), 'constant', constant_values=pad_val)
    return out


class TooLongError(Exception):
    pass


def idxs2np(idxs, size, pad_val=1):
    out = pad(idxs, size, pad_val=pad_val)
    return out


def get_spans(text, tokens):
    """

    :param text:
    :param tokens:
    :return: a list of char-level spans, where each span is an exclusive range, i.e. (start, stop)
    """
    cur_idx = 0
    spans = []
    for token in tokens:
        cur_idx = text.find(token, cur_idx)
        assert cur_idx >= 0, "Text and tokens do not match."
        spans.append((cur_idx, cur_idx + len(token)))
        cur_idx += len(token)
    return spans


def get_word_span(context, words, start, stop):
    spans = get_spans(context, words)
    idxs = []
    for word_idx, span in enumerate(spans):
        if not (stop <= span[0] or start >= span[1]):
            idxs.append(word_idx)
    assert len(idxs) > 0, "context and words do not match, or start and stop are not valid indices."
    return idxs[0], idxs[-1]


def get_phrase(context, words, span):
    start, stop = span
    char_idx = 0
    char_start, char_stop = None, None
    for word_idx, word in enumerate(words):
        char_idx = context.find(word, char_idx)
        assert char_idx >= 0
        if word_idx == start:
            char_start = char_idx
        char_idx += len(word)
        if word_idx == stop:
            char_stop = char_idx
            break
    assert char_start is not None
    assert char_stop is not None, (context, words, span, words[span[0]:span[1]+1])
    phrase = context[char_start:char_stop]
    return phrase


def get_best_span(yp1, yp2, op=None):
    max_val = -10e9
    best_word_span = None
    best_start_index = 0
    if op is None:
        op = operator.mul
    for j in range(len(yp1)):
        val1 = yp1[best_start_index]
        if val1 <= yp1[j]:
            val1 = yp1[j]
            best_start_index = j

        val2 = yp2[j]
        if op(val1, val2) >= max_val:
            best_word_span = (best_start_index, j)
            max_val = op(val1, val2)

    assert best_word_span is not None
    assert best_start_index is not None
    return best_word_span, float(max_val)


def get_word2vec(glove_path, num_words=400000, draft=False):
    word2vec_dict = {}
    with open(glove_path, 'r') as fp:
        for idx, line in tqdm(enumerate(fp), total=num_words, desc="get_word2vec"):
            tokens = line.strip().split(" ")
            word = tokens[0]
            vec = list(map(float, tokens[1:]))
            word2vec_dict[word] = vec
            if draft and idx + 1 >= num_words / 100:
                break

    return word2vec_dict
