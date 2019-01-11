import os
import datetime
from data import datasets
import logging
import sys
from nets import Sam1, Sam2, Sam3
import multiprocessing as mp
import argparse

parser = argparse.ArgumentParser(
    prog='Code to train and evaluate a set of networks on a data.')
parser.add_argument('-C', '--config', required=True)

NETS = [Sam1.SAM1, Sam2.SAM2, Sam3.SAM3]
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
    features.set_shape([None, 26])
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.export.TensorServingInputReceiver(features=features,
                                                        receiver_tensors=features)
    return features, labels




def create_model_fn(features, labels, mode, params):
    import tensorflow as tf
    print(params['index'])
    network = NETS[params['index']](params['config'].training.numclasses)
    output = network.logits(features)
    if mode == tf.estimator.ModeKeys.TRAIN:
        loss = tf.losses.mean_squared_error(labels=labels, predictions=output)
        loss = tf.reduce_mean(loss)
        opt = tf.train.AdamOptimizer(learning_rate=params['config'].training.learning_rate)
        train_op = opt.minimize(loss, global_step=tf.train.get_global_step())
        predictions = output
        accuracy = loss
        logging_hook = tf.train.LoggingTensorHook({"loss" : loss,
                                                   "MSE Error" : accuracy}, every_n_iter=params['config'].training.display_steps)
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.TRAIN,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[logging_hook],
                                          eval_metric_ops=None)

    if mode == tf.estimator.ModeKeys.EVAL:
        loss = tf.losses.mean_squared_error(labels=labels, predictions=output)

        loss = tf.reduce_mean(loss)
        predictions = output
        accuracy = loss
        logging_hook = tf.train.LoggingTensorHook({"loss": loss,
                                                   "MSE Error" : accuracy}, every_n_iter=params['config'].training.display_steps)
        return tf.estimator.EstimatorSpec(mode=tf.estimator.ModeKeys.EVAL,
                                          loss=loss,
                                          evaluation_hooks=[logging_hook],
                                          eval_metric_ops=None)

    if mode == tf.estimator.ModeKeys.PREDICT:
        export_outputs = {'output' : tf.estimator.export.PredictOutput({'temperature_C1' : output[0],
                          'tempaerature_C2' : output[1],
                          'humidity_C1' : output[2],
                          'humidity_C2' : output[3]})}

        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=output,
                                          export_outputs=export_outputs)


def build_estimator(config):
    cpu_name = mp.current_process().name
    cpu_id = int(cpu_name[cpu_name.find('-') + 1:]) - 1
    gpu_id = gpu_id_list[cpu_id]
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_id)
    import tensorflow as tf
    tf.logging.set_verbosity(tf.logging.INFO)
    train_input_fn = lambda : create_input_fn(config, tf.estimator.ModeKeys.TRAIN)
    eval_input_fn = lambda : create_input_fn(config, tf.estimator.ModeKeys.EVAL)
    predict_input_fn = lambda : create_input_fn(config, tf.estimator.ModeKeys.PREDICT)
    params = {
        'index' : gpu_id,
        'config' : config
    }
    print(gpu_id)
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
    estimator.export_saved_model(os.path.join(exp_folder, 'final_model'), predict_input_fn)
    return None


if __name__ == "__main__":
    args = parser.parse_args()
    config_file = args.config
    training_config = parse_config(config_file)
    print(training_config)
    gpu_id_list = list(range(0,training_config.training.numgpus))
    if len(NETS) <= len(gpu_id_list):
        numproc = len(NETS)
    else:
        numproc = len(gpu_id_list)
    pool = mp.Pool(processes=numproc)
    mapargs = [training_config] * len(NETS)
    pool.map(build_estimator, mapargs)

