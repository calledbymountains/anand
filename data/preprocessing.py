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

'''
def preprocess_train(data):
    data = parse_proto(data)
    image = tf.image.decode_image(data['image'])
    image = tf.cast(image, tf.float32)
    height = tf.cast(data['height'], tf.int32)
    width = tf.cast(data['width'], tf.int32)
    image = tf.reshape(image, [height, width, 3])
    label = data['label_number']
    label = tf.one_hot(label, 10)
    label = tf.cast(label, tf.float32)
    return image, label
'''

'''
def preprocess_val(data):
    data = parse_proto(data)
    image = tf.image.decode_image(data['image'])
    image = tf.cast(image, tf.float32)
    height = tf.cast(data['height'], tf.int32)
    width = tf.cast(data['width'], tf.int32)
    image = tf.reshape(image, [height, width, 3])
    label = data['label_number']
    label = tf.one_hot(label, 10)
    label = tf.cast(label, tf.float32)
    return image, label
    
'''


def preprocess_train(data):
    line_split = tf.strings.split([data], ',')
    parsed_data = {
        'summary' : tf.cast(tf.strings.to_number(line_split.values[0], tf.float32), tf.uint8),
        'precipIntensity' : tf.strings.to_number(line_split.values[1], tf.float32),
        'precipProbability' : tf.strings.to_number(line_split.values[2], tf.float32),
        'temperature' : tf.strings.to_number(line_split.values[3], tf.float32),
        'temperature_apparent' : tf.strings.to_number(line_split.values[4], tf.float32),
        'dewpoint' : tf.strings.to_number(line_split.values[5], tf.float32),
        'humidity' : tf.strings.to_number(line_split.values[6], tf.float32),
        'windspeed' : tf.strings.to_number(line_split.values[7], tf.float32),
        'windbearing' : tf.strings.to_number(line_split.values[8], tf.float32),
        'visibility' : tf.strings.to_number(line_split.values[9], tf.float32),
        'cloudcover' : tf.strings.to_number(line_split.values[10], tf.float32),
        'pressure' : tf.strings.to_number(line_split.values[11], tf.float32),
        'dewpoint1' : tf.strings.to_number(line_split.values[12], tf.float32),
        'dewpoint2' : tf.strings.to_number(line_split.values[13], tf.float32),
        'thermostat' : tf.strings.to_number(line_split.values[14], tf.float32),
        'desired_temp' : tf.strings.to_number(line_split.values[15], tf.float32),
        'temperature_C1' : tf.strings.to_number(line_split.values[16], tf.float32),
        'temperature_C2' : tf.strings.to_number(line_split.values[17], tf.float32),
        'humidity_C1' : tf.strings.to_number(line_split.values[18], tf.float32),
        'humidity_C2' : tf.strings.to_number(line_split.values[19], tf.float32)
    }

    features = tf.stack([
                         parsed_data['precipIntensity'],
                         parsed_data['precipProbability'],
                         parsed_data['temperature'],
                         parsed_data['temperature_apparent'],
                         parsed_data['dewpoint'],
                         parsed_data['humidity'],
                         parsed_data['windspeed'],
                         parsed_data['windbearing'],
                         parsed_data['visibility'],
                         parsed_data['cloudcover'],
                         parsed_data['pressure'],
                         parsed_data['dewpoint1'],
                         parsed_data['dewpoint2'],
                         parsed_data['thermostat'],
                         parsed_data['desired_temp']])
    features = tf.concat([tf.one_hot(parsed_data['summary'], depth=11), features], axis=0)
    targets  = tf.stack([parsed_data['temperature_C1'],
                         parsed_data['temperature_C2'],
                         parsed_data['humidity_C1'],
                         parsed_data['humidity_C2']])
    return features, targets

def preprocess_val(data):
    line_split = tf.strings.split([data], ',')

    parsed_data = {
        'summary': tf.cast(tf.strings.to_number(line_split.values[0], tf.float32), tf.uint8),
        'precipIntensity' : tf.strings.to_number(line_split.values[1], tf.float32),
        'precipProbability' : tf.strings.to_number(line_split.values[2], tf.float32),
        'temperature' : tf.strings.to_number(line_split.values[3], tf.float32),
        'temperature_apparent' : tf.strings.to_number(line_split.values[4], tf.float32),
        'dewpoint' : tf.strings.to_number(line_split.values[5], tf.float32),
        'humidity' : tf.strings.to_number(line_split.values[6], tf.float32),
        'windspeed' : tf.strings.to_number(line_split.values[7], tf.float32),
        'windbearing' : tf.strings.to_number(line_split.values[8], tf.float32),
        'visibility' : tf.strings.to_number(line_split.values[9], tf.float32),
        'cloudcover' : tf.strings.to_number(line_split.values[10], tf.float32),
        'pressure' : tf.strings.to_number(line_split.values[11], tf.float32),
        'dewpoint1' : tf.strings.to_number(line_split.values[12], tf.float32),
        'dewpoint2' : tf.strings.to_number(line_split.values[13], tf.float32),
        'thermostat' : tf.strings.to_number(line_split.values[14], tf.float32),
        'desired_temp' : tf.strings.to_number(line_split.values[15], tf.float32),
        'temperature_C1' : tf.strings.to_number(line_split.values[16], tf.float32),
        'temperature_C2' : tf.strings.to_number(line_split.values[17], tf.float32),
        'humidity_C1' : tf.strings.to_number(line_split.values[18], tf.float32),
        'humidity_C2' : tf.strings.to_number(line_split.values[19], tf.float32)
    }
    features = tf.stack([parsed_data['precipIntensity'],
                         parsed_data['precipProbability'],
                         parsed_data['temperature'],
                         parsed_data['temperature_apparent'],
                         parsed_data['dewpoint'],
                         parsed_data['humidity'],
                         parsed_data['windspeed'],
                         parsed_data['windbearing'],
                         parsed_data['visibility'],
                         parsed_data['cloudcover'],
                         parsed_data['pressure'],
                         parsed_data['dewpoint1'],
                         parsed_data['dewpoint2'],
                         parsed_data['thermostat'],
                         parsed_data['desired_temp']])
    features = tf.concat([tf.one_hot(parsed_data['summary'], depth=11), features], axis=0)
    targets  = tf.stack([parsed_data['temperature_C1'],
                         parsed_data['temperature_C2'],
                         parsed_data['humidity_C1'],
                         parsed_data['humidity_C2']])
    return features, targets


