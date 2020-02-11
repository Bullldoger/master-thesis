"""

"""


from _collections import OrderedDict
import random


class BasicNet:
    """

    """

    def __init__(self, *args, **kwargs):
        """

        """
        self.results = OrderedDict()
        self.y = None
        self.x = None
        self.layers = OrderedDict()

        for layer in args:
            layer_name = layer.name
            self.layers[layer_name] = layer

        self.name = kwargs.get('name', str(random.randint(0, 2 ** 5)))

    def forward(self, x, *args, **kwargs):
        """

        """

        self.x = x
        y = x
        self.results['input'] = {
            'x': y
        }

        for layer_name, layer in self.layers.items():
            y = layer.forward(y, *args, **kwargs)

            self.results[layer_name] = {
                'x': y
            }

        self.y = y
        return self.y

    def __str__(self):

        printable = f'Net {self.name}:'
        template = '\n\t{}'
        for layer_name, layer in self.layers.items():
            printable += template.format(layer.__str__())

        return printable
