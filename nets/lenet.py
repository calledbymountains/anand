from nets.basenetwork import BaseNet
import tensorflow as tf


class LeNet(BaseNet):
    def __init__(self, input_batch, num_classes, input_shape=(28, 28, 3),
                 pretrained_model=None):
        super(LeNet, self).__init__('LeNet', num_classes, input_shape,
                                    pretrained_model)
        self._logits = self.logits(input_batch)

    def logits(self, input_batch):
        output = tf.layers.conv2d(inputs=input_batch,
                                  filters=32,
                                  kernel_size=5,
                                  activation=tf.nn.relu,
                                  kernel_initialier=tf.initializers.glorot_normal,
                                  name='conv1')

        output = tf.layers.max_pooling2d(inputs=output,
                                         pool_size=2,
                                         strides=2,
                                         name='pool1')

        output = tf.layers.conv2d(inputs=input_batch,
                                  filters=64,
                                  kernel_size=5,
                                  activation=tf.nn.relu,
                                  kernel_initialier=tf.initializers.glorot_normal,
                                  name='conv2')

        output = tf.layers.max_pooling2d(inputs=output,
                                         pool_size=2,
                                         strides=2,
                                         name='pool2')

        output = tf.layers.flatten(output, name='flatten')

        output = tf.layers.dense(inputs=output, units=1024,
                                 kernel_initializer=tf.initializers.glorot_normal,
                                 name='fc1')

        output = tf.layers.dropout(output, rate=0.5)

        output = tf.layers.dense(inputs=output, units=self.num_classes,
                                 kernel_initializer=tf.initializers.glorot_normal,
                                 name='fc2')

        return output

    def input_shape(self):
        return (28, 28, 3)
