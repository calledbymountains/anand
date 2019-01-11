from nets.basenetwork import BaseNet
import tensorflow as tf


class SAM1(BaseNet):
    def __init__(self, pretrained_model=None):
        super(SAM1, self).__init__('SAM1', num_classes=4, input_shape=26, pretrained_model=pretrained_model)

    def logits(self, input_batch):
        layer1 = tf.layers.dense(inputs=input_batch, units=8, kernel_initializer=tf.initializers.glorot_normal, activation=tf.nn.sigmoid,
                                 name='h1')


        layer2 = tf.layers.dense(inputs=layer1, units=16,
                                 kernel_initializer=tf.initializers.glorot_normal, activation=tf.nn.sigmoid,
                                 name='h2')

        layer3 = tf.layers.dense(inputs=layer2, units=16,
                                 kernel_initializer=tf.initializers.glorot_normal, activation = tf.nn.sigmoid,
                                 name='h3')

        output = tf.layers.dense(inputs=layer3, units=self.num_classes, kernel_initializer=tf.initializers.glorot_normal, activation=None,
                                 name='output')

        return output

    def input_shape(self):
        return 26


