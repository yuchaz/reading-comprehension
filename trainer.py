import tensorflow as tf


def get_train_op(config, losses, scope=None):
    with tf.name_scope(scope or "optimization"):
        opt = tf.train.AdadeltaOptimizer(config.init_lr)
        grads_list = []

        for device_idx in range(1):
            with tf.name_scope("grads_{}".format(device_idx)), tf.device("/{}:{}".format(config.device_type, device_idx)):
                grads = opt.compute_gradients(losses[device_idx])
                grads_list.append(grads)

        avg_grads = grads_list[0]  # average_gradients(grads_list)
        train_op = opt.apply_gradients(avg_grads, global_step=tf.train.get_global_step())
        return train_op


