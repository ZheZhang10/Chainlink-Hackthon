import tensorflow as tf
import typing
from helpful_scripts import CONTENT_LAYERS, STYLE_LAYERS


def get_vgg19_model(layers):
    # build and init the vgg19 model
    # load the model
    vgg = tf.keras.applications.VGG19(include_top=False, weights="imagenet")
    # extract the output of vgg19's layers will be used
    outputs = [vgg.get_layer(layer).output for layer in layers]
    # build new model with output
    model = tf.keras.Model([vgg.input,], outputs)
    # lock the parameters, not training
    model.training = False
    return model


class NeuralStyleTransferModel(tf.keras.Model):
    def __init__(
        self,
        content_layers: typing.Dict[str, float] = CONTENT_LAYERS,
        style_layers: typing.Dict[str, float] = STYLE_LAYERS,
    ):
        super(NeuralStyleTransferModel, self).__init__()
        # content feature layer Dict[layer name, weight]
        self.content_layers = content_layers
        # style feature layer
        self.style_layers = style_layers
        # extract all need vgg layer
        layers = list(self.content_layers.keys()) + list(self.style_layers.keys())
        # mapping of layer name and output
        self.outputs_index_map = dict(zip(layers, range(len(layers))))
        # creat and init vgg network
        self.vgg = get_vgg19_model(layers)

    def call(self, inputs, training=None, mask=None):
        """
    forward propagation
    """
        outputs = self.vgg(inputs)
        # separate the output of content feature layer and style feature layer,
        # whcih is convenient for calculating typing.List[outputs, weights]
        content_outputs = []
        for layer, factor in self.content_layers.items():
            print(layer)
            print(factor)
            content_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
            # print(self.outputs_index_map[layer][0])
        style_outputs = []
        for layer, factor in self.style_layers.items():
            style_outputs.append((outputs[self.outputs_index_map[layer]][0], factor))
        # return the output as dictory
        return {"content": content_outputs, "style": style_outputs}

