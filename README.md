# Documentation


## Requirements

Please ensure that the following packages are installed
1. TensorFlow : 1.10-1.12 in Python 3.6
2. OpenCV 3.4.0
3. Protobuf 

## Usage

### Installation

After ensuring that the above requirements are satisfied, please run `sh compile_proto.sh` to compile the proto files on your system.
### Data Preparation

To prepare the data, please use the `utils/prepare_tfrecords.py` file. 

You need to create 3 text files : `imagelist.txt`, `labellist.txt` and `labelmap.txt`

1. `imagelist.txt` should contain the full path to your image files ( one each line).
2. `labellist.txt` should contain the integer labels for each image file ( must be 0-indexed). Moreover, each image should have exactly one label. 

**Note** : The ordering between `imagelist.txt` and `labellist.txt` must exactly match.

3. `labelmap.txt` should contain the mapping between an integer label and the corresponding class name, separated by a tab. An example is as follows :
```
0    Bear
1    Cat
3    Dog
```

After creating these text files run the code as follows :

`python utils/prepare_tfrecords.py --image_file <PATH to imagelist.txt> --label_file <PATH to labellist.txt> --label_map <PATH to labelmap.txt> --num_shards <Number of shards to create> --save_path <Path where the files should be saved> --save_name <Name of tfrecords>`

**Preparation of the configuration file**

A sample configuration file has been provided ( `sample_config.config`). It is in protobuf text format.
Please feel free to create your own file after this format. A lot of other options are also supported, which have default values and do not appear in the `sample_config.config` file. To get a full understanding of those options, please see the `.proto` files in the `protos` folder.

Many of the options in the config file reflect standard tensorflow options.

**Code Usage** 

**NOTE**: Specify the `numgpus` option in the config file carefully. If you have only 16 GPUs on one node, and specify `numgpus` as `32`, the code would fail to run.

Run the code as follows :

`python train_and_evaluate.py --config <PATH to your config file>`

## Code Structure

1. `data` : Contains the code to prepare the dataset and the preprocessing. If you want to add your own preprocessing functions, please edit `data/preprocessing.py`.
2. `nets` : Contains the code to prepare a network. If you want to add a network of your own, please follow `nets/basenetwork.py` to see the abstract class from which all networks should be inherited. An example is in `nets/lenet.py`. This uniform interfacing is important so that one function can initialize all the networks.
3. `protos` : Contains the protobuf files.
4. `utils` : The only file of importance here is the `prepare_tfrecords.py`, which has been described before.
5. `compile_proto.sh` : Is a small bash script to compile the proto files.
6. `train_and_evaluate.py` : Is the main code for training and evaluation.

