import itertools
import time
import threading

import tensorflow as tf

from data import Data


class InputQueue(object):
    def __init__(self, config, data, names=None):
        assert isinstance(data, Data)
        self._data = data
        self._names = names or data.names
        self._cycle = config.train
        self._shapes = [data.shapes[name] for name in self._names]
        self._dtypes = [data.dtypes[name] for name in self._names]
        if config.train:
            self._queue = tf.RandomShuffleQueue(capacity=config.queue_capacity,
                                                min_after_dequeue=config.min_after_dequeue,
                                                shapes=self._shapes,
                                                dtypes=self._dtypes)
        else:
            self._queue = tf.FIFOQueue(capacity=config.queue_capacity,
                                       shapes=self._shapes,
                                       dtypes=self._dtypes)
        self._queue_size = self._queue.size()

        self._placeholders_list = [{name: tf.placeholder(data.dtypes[name], data.shapes[name], name)
                                    for name in self._names} for idx in range(config.num_devices)]
        self._enqueue_op = self._queue.enqueue([placeholders[name]
                                                for placeholders in self._placeholders_list
                                                for name in self._names])
        input_list = self._queue.dequeue_many(config.batch_size_ph)
        # FIXME : split will fail if batch size cannot be divided into num_devices.
        input_list_split = zip(*[tf.split(input_, config.num_devices, axis=0) for input_ in input_list])
        self.inputs_list = [dict(zip(self._names, input_list)) for input_list in input_list_split]
        self.inputs = dict(zip(self._names, input_list))
        self.threads = []

    def _thread_main(self, sess, coord):
        assert isinstance(sess, tf.Session)
        it = itertools.cycle(self._data) if self._cycle else self._data
        for each in it:
            if coord.should_stop():
                break
            feed_dict = {placeholders[name]: each[name] for placeholders in self._placeholders_list for name in self._names}
            sess.run(self._enqueue_op, feed_dict=feed_dict)

    def start(self, sess, coord):
        for thread in self.threads:
            assert not thread.is_alive()
        self.threads = [threading.Thread(target=self._thread_main, args=(sess, coord)) for _ in range(1)]
        # tf.train.start_queue_runners(sess=sess, coord=self._coord)
        for thread in self.threads:
            thread.start()

    def join(self, coord):
        coord.join(self.threads)

    def get_queue_size(self, sess):
        return sess.run(self._queue_size)

    # TODO : Enable length-aware batching for faster training.
    # Load X number of examples into queue, and sort them within the queue

