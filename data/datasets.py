import tensorflow as tf
from data.preprocessing import preprocess_train, preprocess_val
from protos import data_pb2

def db_train(proto_config):
    if not proto_config is data_pb2.data_proto:
        raise ValueError('The proto config specified is not valid. Please see protos/data.proto to see the correct configuration.')

    tfrecords = tf.gfile.Glob(proto_config.tfrecord_list_glob)
    num_parallel_calls = proto_config.num_parallel_reads
    read_buffer_size = proto_config.read_buffer_size
    db = tf.data.TFRecordDataset(filenames=tfrecords,
                                 num_parallel_reads=num_parallel_calls,
                                 buffer_size=read_buffer_size)

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

    dbiter = db.make_initializable_iterator()
    return dbiter


def db_val(proto_config):
    if not proto_config is data_pb2.data_proto:
        raise ValueError(
            'The proto config specified is not valid. Please see protos/data.proto to see the correct configuration.')

    tfrecords = tf.gfile.Glob(proto_config.tfrecord_list_glob)
    num_parallel_calls = proto_config.num_parallel_reads
    read_buffer_size = proto_config.read_buffer_size
    db = tf.data.TFRecordDataset(filenames=tfrecords,
                                 num_parallel_reads=num_parallel_calls,
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

    dbiter = db.make_initializable_iterator()
    return dbiter