"""

"""

import pden
from pden import Net
from pden import Operations

import tensorflow as tf
import tqdm
import time


class HadamardNET:
    """

    """

    def __init__(self, *args, **kwargs):
        """

        """
        self.learning_rate = kwargs.get('learning_rate', 1e-3)
        self.training_epochs = kwargs.get('training_epochs', 200)

        self.loss = 0
        self.loss_factors = []
        self.y = None
        self.x = None
        self.sess = None

    def train(self, feed_dict: dict, reset=True):

        opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train = opt.minimize(self.loss)
        init = tf.global_variables_initializer()

        if self.sess is None or reset:
            self.sess = tf.Session()
            self.sess.run(init)

        loss = None
        t_start = time.clock()
        for _ in tqdm.tqdm(range(self.training_epochs), total=self.training_epochs, desc='Training progress'):
            _, loss, y = self.sess.run([train, self.loss, self.y], feed_dict)

        print(f'Training finished in: \t{time.clock() - t_start}')
        print(f'\tAfter {self.training_epochs}, loss is {loss}')

        return y

    def add_loss(self, loss, weight=1.0):
        """

        """
        self.loss_factors += [loss]
        self.loss += weight * loss

        return self