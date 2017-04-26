import os
import numpy as np

import tensorflow as tf

from dcgan.model import DCGAN
from config import get_config

flags = tf.app.flags
FLAGS = flags.FLAGS

def main(_):
# gpu option
    with tf.Session() as sess:
        config = get_config(FLAGS) or FLAGS

        dcgan = DCGAN(sess, config)
        dcgan.train()

if __name__ == '__main__':
    tf.app.run()
