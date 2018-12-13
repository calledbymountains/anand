import os
import tensorflow as tf
from protos import training_pb2
from data import datasets
from google.protobuf import text_format
from nets import lenet
import argparse

parser = argparse.ArgumentParser(
    prog='Code to train and evaluate a set of networks on a data.')
parser.add_argument('-C', '--config', required=True)

NETS = [lenet.LeNet] * 20


def parse_config(config_file):
    if not os.path.exists(config_file):
        raise ValueError(
            'The config file {} does not exist.'.format(config_file))

    with open(config_file, 'r') as f:
        proto_str = f.read()
        training_config = training_pb2.Training()
        text_format.Merge(proto_str, training_config)

    return training_config

def get_next_data(iterator):
    return iterator.get_next()

def create_model(network, config)
    if config is not training_pb2.Training:
        raise ValueError('The proto configuration is not correct. Please check protos/training.proto to see the correct configuration format.')

    dbtrain = datasets.db_train(config.data_info.train_data)
    dbval = datasets.db_val(config.data_info.val_data)
    phase = tf.get_variable('phase', shape=(), dtype=tf.bool, trainable=False)
    data = tf.cond(phase, lambda : get_next_data(dbtrain), lambda: get_next_data(dbval))
    net = network(data, )





if __name__ == "__main__":
    args = parser.parse_args()
    config_file = args.config
    training_config = parse_config(config_file)
    print(training_config)
