"""

"""

import pden
from pden import Net
from pden import Operations

import tensorflow as tf
import tqdm
import time


class BasicNET:
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

    def derivatives(self):
        """

        """
        ys = tf.split(self.y, [1] * self.dimension_out, 1)

        def _derivatives(i, j=[]):
            f = ys[i]
            for _j in j:
                fs = tf.gradients(f, self.net.x)[0]
                f = tf.split(fs, [1] * self.dimension_in, 1)[_j]
            return f

        return _derivatives


class PDENET(BasicNET):
    """

    """

    def __init__(self, dimension_in=1, dimension_out=1, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.net = kwargs.get('net', Net.BasicNet(*args, **kwargs))
        self.dimension_in = dimension_in
        self.dimension_out = dimension_out

    def add_loss(self, loss, weight=1.0):
        """

        """
        self.loss_factors += [loss]
        self.loss += weight * loss

        return self

    def forward(self, x, *args, **kwargs):
        """

        """
        self.y = self.net.forward(x, *args, **kwargs)
        return self.y


class HadamardNET(BasicNET):
    """

    """

    def __init__(self, dimension_in=1, dimension_out=1, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)

        self.hidden_size = kwargs.get('hidden', 8)
        self.nets = kwargs.get('nets', None)
        self.dimension_in = dimension_in
        self.dimension_out = dimension_out

    def add_loss(self, loss, weight=1.0):
        """

        """
        self.loss_factors += [loss]
        self.loss += weight * loss

        return self

    def forward(self, x, *args, **kwargs):
        """

        """
        xs = tf.split(x, [1] * self.dimension_in, axis=1)
        ys = [n.forward(_x, *args, **kwargs) for _x, n in zip(xs, self.nets)]
        self.y = ys[0]
        for _y in ys[1:]:
            self.y = tf.multiply(self.y, _y)

        self.y = self.nets[-1].forward(self.y)
        return self.y


class FourierNet1D(BasicNET):
    """

    """

    def __init__(self, fourier_terms=2, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)
        self.net_cos = pden.Net.BasicNet(
            pden.Operations.Linear(feature_out=fourier_terms, random_init=True),
            pden.Operations.ActivationFunction(tf.cos),
            pden.Operations.Linear(feature_out=1, feature_in=fourier_terms, random_init=True), **kwargs
        )

        self.net_sin = pden.Net.BasicNet(
            pden.Operations.Linear(feature_out=fourier_terms, random_init=True),
            pden.Operations.ActivationFunction(tf.sin),
            pden.Operations.Linear(feature_out=1, feature_in=fourier_terms, random_init=True), **kwargs
        )

        self.dimension_out = 1
        self.dimension_in = 2

    def add_loss(self, loss, weight=1.0):
        """

        """
        self.loss_factors += [loss]
        self.loss += weight * loss

        return self

    def forward(self, x, *args, **kwargs):
        """

        """
        y_cos = self.net_cos.forward(x, *args, **kwargs)
        y_sin = self.net_sin.forward(x, *args, **kwargs)

        self.y = y_cos + y_sin

        return self.y


class FourierNet2D(BasicNET):
    """

    """

    def __init__(self, fourier_terms=2, *args, **kwargs):
        """

        """
        super().__init__(*args, **kwargs)

        self.dimension_out = 1
        self.dimension_in = 2
        self.fourier_terms = fourier_terms
        self.nets = [
            self._net_generator(terms=fourier_terms) for _ in range(2)
        ]

    @staticmethod
    def net_generator(terms=2):
        """

        """
        net_cos = pden.Net.BasicNet(
            pden.Operations.Linear(feature_out=terms, random_init=True),
            pden.Operations.ActivationFunction(tf.cos),
            pden.Operations.Linear(feature_out=1, feature_in=self.fourier_terms, random_init=True), **kwargs
        )

        net_sin = pden.Net.BasicNet(
            pden.Operations.Linear(feature_out=terms, random_init=True),
            pden.Operations.ActivationFunction(tf.sin),
            pden.Operations.Linear(feature_out=1, feature_in=fourier_terms, random_init=True), **kwargs
        )

        return net_cos, net_sin

    def add_loss(self, loss, weight=1.0):
        """

        """
        self.loss_factors += [loss]
        self.loss += weight * loss

        return self

    def forward(self, x, *args, **kwargs):
        """

        """
        y = None

        for net_sin, net_cos in self.nets:
            y_cos = net_cos.forward(x, *args, **kwargs)
            y_sin = net_sin.forward(x, *args, **kwargs)
            y = y * (y_cos + y_sin) if y is not None else y_cos + y_sin

        self.y = y
        return self.y
