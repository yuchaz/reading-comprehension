import operator

import numpy as np
import tensorflow as tf

from data import get_best_span, get_phrase
from evaluate import evaluate


class Evaluation(object):
    pass


class SquadEvaluation(Evaluation):
    def __init__(self, data, inputs=None, outputs=None, loss=None, global_step=None):
        self.inputs = inputs or {}
        self.outputs = outputs or {}
        self.global_step = global_step
        self.data = data
        self.loss = loss
        self.acc = None
        self.score = None

    @property
    def num_examples(self):
        if len(self.inputs) == 0:
            return 0
        return len(self.inputs['q'])

    def __add__(self, other):
        assert isinstance(other, SquadEvaluation)
        if self.loss is None or other.loss is None:
            loss = None
        else:
            if self.num_examples + other.num_examples == 0:
                loss = 0
            else:
                loss = (self.loss * self.num_examples + other.loss * other.num_examples) / (self.num_examples + other.num_examples)
        global_step = self.global_step or other.global_step
        inputs, outputs = {}, {}
        if other.inputs is not None:
            for key, vals in other.inputs.items():
                if key in self.inputs:
                    inputs[key] = np.append(self.inputs[key], vals, axis=0)
                else:
                    inputs[key] = vals
        if other.outputs is not None:
            for key, vals in other.outputs.items():
                if key in self.outputs:
                    outputs[key] = np.append(self.outputs[key], vals, axis=0)
                else:
                    outputs[key] = vals
        return SquadEvaluation(self.data, inputs=inputs, outputs=outputs, loss=loss, global_step=global_step)

    def __repr__(self):
        acc1, acc2 = self.get_acc()['acc1'], self.get_acc()['acc2']
        em, f1 = self.get_score()['exact_match'], self.get_score()['f1']
        return str('<{} at {}> loss: {:.4f}, acc1: {:.3f}%, acc2: {:.3f}%, EM: {:.3f}%, F1: {:.3f}%'.format(self.data.data_type, self.global_step, self.loss, acc1, acc2, em, f1))

    def get_answers(self):
        idxs = self.inputs['idxs']
        logits1_list, logits2_list = self.outputs['logits1'], self.outputs['logits2']
        answers = {}
        for idx, logits1, logits2 in zip(idxs, logits1_list, logits2_list):
            each = self.data.get(idx)
            context, context_words, id_ = [each[key] for key in ['context', 'context_words', 'ids']]
            best_span, best_score = get_best_span(logits1, logits2, op=operator.add)
            # rx = self.data.data['*x'][idx]
            # context, context_words = self.data.shared['context'][rx], self.data.shared['context_words'][rx]
            answer = get_phrase(context, context_words, best_span)
            id_ = each['ids']
            answers[id_] = answer
        return answers

    def get_score(self):
        if self.score is not None:
            return self.score
        answers = self.get_answers()
        official = evaluate(self.data.squad['data'], answers)
        self.score = official
        return official

    def get_acc(self):
        if self.acc is not None:
            return self.acc
        y1, y2 = self.inputs['y1'], self.inputs['y2']  # [N]
        yp1, yp2 = self.outputs['yp1'], self.outputs['yp2']  # [N]
        acc1 = 100 * np.mean(np.equal(y1, yp1))
        acc2 = 100 * np.mean(np.equal(y2, yp2))
        acc = {'acc1': acc1, 'acc2': acc2}
        self.acc = acc
        return acc

    def get_summaries(self):
        acc = self.get_acc()
        score = self.get_score()
        acc1, acc2 = acc['acc1'], acc['acc2']
        em, f1 = score['exact_match'], score['f1']
        loss = self.loss
        data_type = self.data.config.data_type
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=loss)])
        acc1_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/acc1'.format(data_type), simple_value=acc1)])
        acc2_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/acc2'.format(data_type), simple_value=acc2)])
        em_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/em'.format(data_type), simple_value=em)])
        f1_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/f1'.format(data_type), simple_value=f1)])
        summaries = [loss_summary, acc1_summary, acc2_summary, em_summary, f1_summary]
        return summaries
