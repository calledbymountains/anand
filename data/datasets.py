import tensorflow as tf
from data.preprocessing import preprocess_train, preprocess_val
from protos import data_pb2

def db_train(proto_config):

    tfrecords = tf.gfile.Glob(proto_config.csv_list_glob)
    read_buffer_size = proto_config.read_buffer_size
    db = tf.data.TextLineDataset(filenames=tfrecords,
                                 buffer_size=read_buffer_size)

    db = db.repeat()
    do_shuffle = proto_config.shuffle

    if do_shuffle:
        shuffle_buffer_size = proto_config.shuffle_buffer_size
        db = db.shuffle(buffer_size=shuffle_buffer_size)

    map_num_parallel_calls = proto_config.map_num_parallel_calls

    db = db.map(preprocess_train, num_parallel_calls=map_num_parallel_calls)

    batchsize = proto_config.batch_size
    db = db.batch(batch_size=batchsize)

    prefetch_buffer_size = proto_config.prefetch_buffer_size

    db = db.prefetch(buffer_size=prefetch_buffer_size)
    return db


def db_val(proto_config):

    tfrecords = tf.gfile.Glob(proto_config.csv_list_glob)
    read_buffer_size = proto_config.read_buffer_size
    db = tf.data.TextLineDataset(filenames=tfrecords,
                                 buffer_size=read_buffer_size)

    do_shuffle = proto_config.shuffle

    if do_shuffle:
        shuffle_buffer_size = proto_config.shuffle_buffer_size
        db = db.shuffle(buffer_size=shuffle_buffer_size)

    map_num_parallel_calls = proto_config.map_num_parallel_calls

    db = db.map(preprocess_val, num_parallel_calls=map_num_parallel_calls)

    batchsize = proto_config.batch_size
    db = db.batch(batch_size=batchsize)

    prefetch_buffer_size = proto_config.prefetch_buffer_size

    db = db.prefetch(buffer_size=prefetch_buffer_size)
    return db