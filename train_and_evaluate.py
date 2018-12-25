import os
import datetime
from data import datasets
import logging
import sys
from nets import lenet
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(
    prog='Code to train and evaluate a set of networks on a data.')
parser.add_argument('-C', '--config', required=True)

NETS = [lenet.LeNet] * 20
gpu_id_list = None



def get_available_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

def parse_config(config_file):
    from google.protobuf import text_format
    from protos import training_pb2
    if not os.path.exists(config_file):
        raise ValueError(
            'The config file {} does not exist.'.format(config_file))

    with open(config_file, 'r') as f:
        proto_str = f.read()
        training_config = training_pb2.Training()
        text_format.Merge(proto_str, training_config)

    return training_config


def create_input_fn(config, mode):
    import tensorflow as tf
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = datasets.db_train(config.training.data_info.train_data)
    else:
        dataset = datasets.db_val(config.training.data_info.val_data)

    dbiter = dataset.make_one_shot_iterator()
    features, labels = dbiter.get_next()
    features.set_shape([None, 28, 28, 3])
    return features, labels


def create_model_fn(features, labels, mode, params):
    import tensorflow as tf
    network = NETS[params['index']](params['config'].training.numclasses)
    output = network.logits(features)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels), logits=output)
        loss = tf.reduce_mean(loss)
        opt = tf.train.MomentumOptimizer(learning_rate=params['config'].training.learning_rate,
                                         momentum=params['config'].training.momentum)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
        predictions = tf.nn.softmax(logits=output)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(labels, axis=1), tf.int64), tf.argmax(predictions, axis=1)), tf.float32))
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss,
                                                   "accuracy" : accuracy}, every_n_iter=params['config'].training.display_steps)
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook],
                                          eval_metric_ops=None)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.stop_gradient(labels),
                                                          logits=output)
        loss = tf.reduce_mean(loss)
        predictions = tf.nn.softmax(logits=output)
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(tf.argmax(labels, axis=1), tf.int64), tf.argmax(predictions, axis=1)),
            tf.float32))
        logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                                   "accuracy" : accuracy}, every_n_iter=params['config'].training.display_steps)
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                          loss=loss,
                                          evaluation_hooks=[logging_hook],
                                          eval_metric_ops=None)


def build_estimator(config):
    cpu_name = mp.current_process().name
    cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
    gpu_id = gpu_id_list[cpu_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id - 1)
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.INFO)
    train_input_fn = lambda : create_input_fn(config, tf.estimator.ModeKeys.TRAIN)
    eval_input_fn = lambda : create_input_fn(config, tf.estimator.ModeKeys.EVAL)
    params = {
        'index' : gpu_id,
        'config' : config
    }
    basepath = config.training.basepath
    currenttime = datetime.datetime.now().strftime('%B-%d-%Y-%I-%M%p')
    exp_folder = os.path.join(basepath, 'gpu-{}-{}'.format(gpu_id, currenttime))
    os.makedirs(exp_folder, exist_ok=True)
    sess_config = tf.ConfigProto(device_count={'GPU': gpu_id})
    cfg = tf.estimator.RunConfig(model_dir=exp_folder,
                                 save_checkpoints_secs=config.training.save_checkpoint_secs, session_config=sess_config)
    estimator = tf.estimator.Estimator(create_model_fn, config=cfg, params=params)
    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=config.training.training_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=config.training.eval_after_steps)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
    return None


if __name__ == "__main__":
    args = parser.parse_args()
    config_file = args.config
    training_config = parse_config(config_file)
    print(training_config)
    gpu_id_list = list(range(1,training_config.training.numgpus+1))
    pool = mp.Pool(processes=len(gpu_id_list))
    mapargs = [training_config] * len(NETS)
    pool.map(build_estimator, mapargs)

