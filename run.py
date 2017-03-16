import os
import shutil
import threading
from collections import defaultdict

import tensorflow as tf
import numpy as np
import time
from tqdm import tqdm

from data import SquadData
from evaluation import SquadEvaluation
from input_queue import InputQueue
# from model import cbow_forward, get_loss, rnn_forward, attention_forward, lstm_forward, lstm_fw_gru_bw, gru_fw_lstm_bw
import model as md
from tf_utils import average_gradients


class Run(object):
    def __init__(self, config, val_config=None):
        self.config = config
        self.val_config = val_config

        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32', trainable=False,
                                           initializer=tf.constant_initializer(0))
        self.opt = tf.train.AdadeltaOptimizer(config.init_lr) if config.is_train else None

        config.batch_size_ph = tf.placeholder("int32", shape=[], name='batch_size')
        config.emb_mat_ph = tf.placeholder("float", shape=[None, config.hidden_size])
        self.data = SquadData(config)
        self.iq = InputQueue(config, self.data)
        self.inputs = self.iq.inputs

        if val_config is not None:
            val_config.batch_size_ph = tf.placeholder("int32", shape=[])
            val_config.emb_mat_ph = tf.placeholder("float", shape=[None, val_config.hidden_size])
            self.val_data = SquadData(val_config, train_data=self.data)
            self.val_iq = InputQueue(val_config, self.val_data)
            self.val_inputs = self.val_iq.inputs

        outputs_list = []
        loss_list = []
        grads_list = []
        with tf.variable_scope("model"):
            with tf.name_scope("train") as train_ns:
                for device_idx in range(config.num_devices):
                    inputs = self.iq.inputs_list[device_idx]
                    with tf.device("/{}:{}".format(config.device_type, device_idx)), \
                            tf.name_scope("{}_{}".format(config.device_type, device_idx)):
                        each_outputs, each_loss, each_grads = self._pipeline(config, inputs)
                        outputs_list.append(each_outputs)
                        loss_list.append(each_loss)
                        grads_list.append(each_grads)
                    if device_idx < config.num_devices - 1:
                        tf.get_variable_scope().reuse_variables()
                self.outputs = self._merge_outputs_list(outputs_list)
                if config.supervised:
                    self.loss = tf.add_n(loss_list)/len(loss_list)
                    if config.is_train:
                        self.grads = grads_list[0]
                        self.train_op = self.opt.apply_gradients(self.grads, global_step=tf.train.get_global_step())
                        self.summary_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES, scope=train_ns))

            if val_config is not None:
                tf.get_variable_scope().reuse_variables()
                with tf.device("/cpu:0"), tf.name_scope("val"):
                    self.val_outputs, self.val_loss, _ = self._pipeline(val_config, self.val_inputs)


        """
        # Adding word embedding metadata
        pc = projector.ProjectorConfig()
        embedding = pc.embeddings.add()
        embedding.tensor_name = self.variables['emb_mat'].name
        embedding.metadata_path = self.config.emb_metadata_path
        projector.visualize_embeddings(self.supervisor.summary_writer, pc)
        """

    def _pipeline(self, config, inputs):
        if config.model == 'cbow':
            model_func = md.cbow_forward
        elif config.model == 'rnn':
            model_func = md.rnn_forward
        elif config.model == 'att':
            model_func = md.attention_forward
        elif config.model == 'lstm':
            model_func = md.lstm_forward
        elif config.model == 'lstm_gru':
            model_func = md.lstm_fw_gru_bw
        elif config.model == 'gru_lstm':
            model_func = md.gru_fw_lstm_bw
        else:
            raise NotImplementedError()
        self.variables, outputs = model_func(config, inputs)
        loss, grads = None, None
        if config.supervised:
            loss = get_loss(config, inputs, outputs)
            if config.is_train:
                grads = self.opt.compute_gradients(loss)
        return outputs, loss, grads

    def _merge_outputs_list(self, outputs_list):
        tmp_outputs = defaultdict(list)
        for each_outputs in outputs_list:
            for key, val in each_outputs.items():
                tmp_outputs[key].append(val)

        outputs = {}
        for key, val_list in tmp_outputs.items():
            outputs[key] = tf.concat(val_list, axis=0)
        return outputs

    def start_queue(self, sess, coord, val=False):
        iq = self.val_iq if val else self.iq
        iq.start(sess, coord)

    def join_queue(self, coord, val=False):
        iq = self.val_iq if val else self.iq
        iq.join(coord)

    def step(self, supervisor, sess, batch_size=None, save_summary=False, val=False):
        config = self.val_config if val else self.config
        inputs = self.val_inputs if val else self.inputs
        outputs = self.val_outputs if val else self.outputs
        data = self.val_data if val else self.data
        loss = self.val_loss if val else self.loss

        batch_size = batch_size or config.batch_size
        input_names = list(inputs)
        input_tensor_list = [inputs[name] for name in input_names]
        output_names = list(outputs)
        output_tensor_list = [outputs[name] for name in output_names]
        args = [input_tensor_list, output_tensor_list, self.global_step]
        if config.supervised:
            args.append(loss)
            if config.is_train:
                args.insert(0, self.train_op)
                if save_summary:
                    args.append(self.summary_op)
        feed_dict = {config.batch_size_ph: batch_size}
        if config.serve:
            feed_dict[config.emb_mat_ph] = config.emb_mat
        results = sess.run(args, feed_dict=feed_dict)

        loss_val = None
        if config.supervised:
            if config.is_train:
                if save_summary:
                    summary = results.pop()
                    supervisor.summary_computed(sess, summary)
                results.pop(0)
            loss_val = results.pop()

        input_val_list, output_val_list, global_step_val = results

        output_vals = dict(zip(output_names, output_val_list))
        input_vals = dict(zip(input_names, input_val_list))
        e = SquadEvaluation(data, inputs=input_vals, outputs=output_vals, loss=loss_val, global_step=global_step_val)
        return e

    def lap(self, supervisor, sess, save=False, val=False):
        data = self.val_data if val else self.data
        config = self.val_config if val else self.config
        assert not config.is_train, "Use step() only for training."

        e = SquadEvaluation(data, loss=0)
        for i in tqdm(range(data.num_batches), desc="{} lap".format(config.data_type)):
            batch_size = config.batch_size if i < data.num_batches - 1 else data.last_batch_size
            each_e = self.step(supervisor, sess, batch_size=batch_size, val=val)
            e += each_e
        if save:
            summaries = e.get_summaries()
            for summary in summaries:
                supervisor.summary_writer.add_summary(summary, sess.run(self.global_step))
        return e


def train(config, val_config=None):
    with tf.device("/cpu:0"):
        if config.fresh:
            if os.path.exists(config.out_dir):
                print("Removing directory: {}".format(config.out_dir))
                shutil.rmtree(config.out_dir, ignore_errors=True)
        if not os.path.exists(config.out_dir):
            print("Creating directory: {}".format(config.out_dir))
            os.makedirs(config.out_dir)

        train_run = Run(config, val_config=val_config)
        supervisor = tf.train.Supervisor(logdir=config.out_dir,
                                         save_model_secs=config.save_model_secs, summary_op=None)
        coord = tf.train.Coordinator()
        th = None
        with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            train_run.start_queue(sess, coord)
            print("Training starts at global_step={}".format(sess.run(train_run.global_step)))
            for step_idx in tqdm(range(config.num_steps), desc="training"):
                if supervisor.should_stop():
                    break
                save_summary = (step_idx + 1) % config.summary_period == 0
                train_run.step(supervisor, sess, save_summary=save_summary)
                if val_config is not None and (step_idx + 1) % config.val_period == 0:
                    if th is not None and th.is_alive():
                        print("Skipping validation at step {}".format(step_idx + 1))
                    else:
                        th = threading.Thread(target=val_routine, args=(tf.get_default_graph(), train_run, supervisor, coord))
                        th.start()

            th.join()

            coord.request_stop()
            train_run.join_queue(coord)
            train_run.join_queue(coord, val=True)


def test(config):
    with tf.device("/cpu:0"):
        test_run = Run(config)

        supervisor = tf.train.Supervisor(logdir=config.out_dir,
                                         save_model_secs=config.save_model_secs, summary_op=None)
        coord = tf.train.Coordinator()

        with supervisor.managed_session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            test_run.start_queue(sess, coord)
            e = test_run.lap(supervisor, sess)
            print(e)

            coord.request_stop()
            test_run.join_queue(coord)


def val_routine(graph, run, train_supervisor, coord):
    with tf.device("/cpu:0"):
        with graph.as_default():
            assert isinstance(run, Run)
            config = run.config
            supervisor = tf.train.Supervisor(logdir=config.out_dir,
                                             save_model_secs=config.save_model_secs, summary_writer=None, summary_op=None)

            with supervisor.managed_session() as sess:
                run.start_queue(sess, coord, val=True)
                e = run.lap(train_supervisor, sess, save=True, val=True)
                print(e)
