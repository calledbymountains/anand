import argparse
import cv2
import multiprocessing as mp
import numpy as np
import os
import random
import tensorflow as tf

parser = argparse.ArgumentParser(
    prog='To create TFRecords for image classification')
parser.add_argument('-I', '--image_file',
                    help='Name of the text file with image paths',
                    required=True)
parser.add_argument('-L', '--label_file',
                    help='Name of the text file with corresponding label numbers.',
                    required=True)
parser.add_argument('-M', '--label_map',
                    help='Name of the text file with the mapping from label numbers to label names.',
                    required=True)
parser.add_argument('-N', '--num_shards', help='Number of shards to create.',
                    type=int, default=100)
parser.add_argument('-S', '--save_path', help='Path to save the TFRecords to',
                    required=True)
parser.add_argument('-Name', '--save_name', help='Name of the TFRecords',
                    required=True)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def prepare_image(image_path):
    if not os.path.exists(image_path):
        raise ValueError(
            'The image {} does not exist. Please check.'.format(image_path))

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ext = os.path.splitext(os.path.basename(image_path))[1]
    height = image.shape[0]
    width = image.shape[1]

    image = cv2.imencode(ext, image)[1].tostring()
    return dict(image=image, height=height, width=width)


def create_example(image_path, label_name, label_number):
    image_dict = prepare_image(image_path)

    example = tf.train.Example(
        features=tf.train.Features(feature={
            'height': _int64_feature(image_dict['height']),
            'width': _int64_feature(image_dict['width']),
            'image': _bytes_feature(image_dict['image']),
            'label_name': _bytes_feature(label_name.encode()),
            'label_number': _int64_feature(label_number)
        })
    )
    return example


def create_record(image_paths, label_names, label_numbers, record_name,
                  shard_num,
                  num_shards, save_path):
    save_name = '{}-{}-of-{}'.format(record_name, str(shard_num).zfill(5),
                                     num_shards)
    final_name = os.path.join(save_path, save_name)
    writer = tf.python_io.TFRecordWriter(final_name)
    for image_path, label_name, label_number in zip(image_paths, label_names,
                                                    label_numbers):
        example = create_example(image_path, label_name, label_number)
        writer.write(example.SerializeToString())
    writer.close()
    return None


def check_existence(file_name):
    if file_name is None:
        raise ValueError('The file name was not provided.')
    if not os.path.exists(file_name):
        raise ValueError('The file {} was not found.'.format(file_name))

    return file_name


def parse_label_map(label_map_file):
    label_dict = dict()
    for line in open(label_map_file, 'r'):
        line = line.strip()
        line = line.split('\t')
        label_num = line[0]
        label_name = line[1]
        label_dict[str(label_num)] = label_name
    return label_dict


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def read_file_to_list(file_name):
    file_list = []
    for line in open(file_name):
        line = line.strip()
        file_list.append(line)
    return file_list


if __name__ == "__main__":
    args = parser.parse_args()
    image_file = check_existence(args.image_file)
    label_file = check_existence(args.label_file)
    label_map = check_existence(args.label_map)
    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    save_name = args.save_name
    num_shards = args.num_shards

    label_dict = parse_label_map(label_map)
    images = read_file_to_list(image_file)
    labels = read_file_to_list(label_file)
    labels = list(map(int, labels))

    if len(labels) != len(images):
        raise ValueError(
            'Number of images and labels is not the same. Please check. Aborting.')

    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)
    label_names = []
    for label in labels:
        label_name = label_dict[str(label)]
        label_names.append(label_name)

    images = chunkIt(images, num_shards)
    labels = chunkIt(labels, num_shards)
    label_names = chunkIt(label_names, num_shards)
    pool = mp.Pool(processes=mp.cpu_count())
    shard_num = np.arange(1, num_shards + 1)
    args = []
    for image, label, label_name, shard in zip(images, labels, label_names,
                                               shard_num):
        args.append(
            tuple((image, label_name, label, save_name, shard, num_shards,
                   save_path)))
    pool.starmap(create_record, args)
    pool.close()
