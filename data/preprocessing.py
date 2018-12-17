import tensorflow as tf


def parse_proto(data):
    features = {
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label_name': tf.FixedLenFeature([], tf.string),
        'label_number': tf.FixedLenFeature([], tf.int64)
    }
    parsed_output = tf.parse_single_example(data, features)
    return parsed_output

def preprocess_train(data):
    data = parse_proto(data)
    image = tf.image.decode_image(data['image'])
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [data['height'], data['width'], 3])
    label = data['label_number']
    label = tf.one_hot(label, 10)
    label = tf.cast(label, tf.float32)
    return image, label

def preprocess_val(data):
    data = parse_proto(data)
    image = tf.image.decode_image(data['image'])
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [data['height'], data['width'], 3])
    label = data['label_number']
    label = tf.one_hot(label, 10)
    label = tf.cast(label, tf.float32)
    return image, label