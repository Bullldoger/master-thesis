"""

"""
import numpy as np
import tensorflow as tf
import random


class BasicOperation:
    """

    """
    def __init__(self, *args, **kwargs):

        self.name = kwargs.get('name', str(random.randint(0, 2 ** 5)))
        pass

    def forward(self, x, *args, **kwargs):
        """

        """
        pass

    def derivative(self, *args, **kwargs):
        """

        """
        pass

    def __str__(self):
        pass


class Linear(BasicOperation):
    """

    """
    def __init__(self, feature_in=1, feature_out=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        assert feature_in > 0
        assert feature_out > 0

        self.feature_in = feature_in
        self.feature_out = feature_out

        self.use_bias = kwargs.get('use_bias', True)
        self.rand_init = kwargs.get('random_init', False)
        self.w = kwargs.get('w', np.zeros((self.feature_in, self.feature_out)) if not self.rand_init else np.random.normal(size=[self.feature_in, self.feature_out]))
        self.b = kwargs.get('b', np.zeros((1, self.feature_out)) if not self.rand_init else np.random.normal(size=[1, self.feature_out])) if self.use_bias else None

        self.W = tf.Variable(self.w, name=f'Linear-{self.name}-W')
        self.B = tf.Variable(self.b, name=f'Linear-{self.name}-B')

        self.y = None

    def forward(self, x, *args, **kwargs):
        """

        """

        self.y = tf.matmul(x, self.W)
        if self.use_bias:
            self.y = tf.add(self.y, self.B)

        return self.y

    def derivative(self, *args, **kwargs):
        pass

    def __str__(self):
        printable = f'{self.name}\tLinear: [{self.feature_in} -> {self.feature_out}]'
        return printable


class ActivationFunction(BasicOperation):
    """

    """
    def __init__(self, func: callable, *args, **kwargs):
        """

        """
        super().__init__(*args, *kwargs)

        assert func is not None

        self.func = func
        self.y = None

    def forward(self, x, *args, **kwargs):
        """

        """
        self.y = self.func(x)
        return self.y

    def derivative(self, *args, **kwargs):
        pass

    def __str__(self):
        printable = f'{self.name}\tActivation funciton: {self.func}'
        return printable
