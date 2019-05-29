import torch
import torch.nn as nn
import warnings

# this ignore is to avoid the flatten parameter warning from showing up. There is no way around it on GPU but it
# still shows up.
warnings.filterwarnings("ignore", category=RuntimeWarning)
"""
This script defines a pytorch deep neural network model for a multi-task classification problem that we 
use in HELEN.

MODEL DESCRIPTION:
The model is a simple bidirectional gated recurrent unit (GRU) based encoder-decoder model. The first layer
performs encoding of the input features and the second layer does a decoder. We use two separate linear layer
to perform the base prediction and RLE prediction. This multi-task method uses a parameter sharing scheme which
is very popular in multi-task classification problems.
"""


class TransducerGRU(nn.Module):
    """
    The GRU based transducer model class.
    """
    def __init__(self, image_channels, image_features, gru_layers, hidden_size, num_base_classes, num_rle_classes,
                 bidirectional=True):
        """
        The initialization of the model
        :param image_channels: Number of channels, usually 1 so not used in this case.
        :param image_features: Features per column in the image
        :param gru_layers: Number of GRU layers in the model
        :param hidden_size: Size of the hidden layer
        :param num_base_classes: Number of classes in base prediction
        :param num_rle_classes: Number of classes in RLE prediction
        :param bidirectional: If true then the GRU layers will be bidirectional.
        """
        super(TransducerGRU, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_layers = gru_layers
        self.num_base_classes = num_base_classes
        self.num_rle_classes = num_rle_classes
        # the gru encoder layer
        self.gru_encoder = nn.GRU(image_features,
                                  hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        # the gru decoder layer
        self.gru_decoder = nn.GRU(2 * hidden_size,
                                  hidden_size,
                                  num_layers=self.num_layers,
                                  bidirectional=bidirectional,
                                  batch_first=True)
        self.gru_encoder.flatten_parameters()
        self.gru_decoder.flatten_parameters()
        # the linear layer for base and RLE classification
        self.dense1_base = nn.Linear(self.hidden_size * 2, self.num_base_classes)
        self.dense2_rle = nn.Linear(self.hidden_size * 2, self.num_rle_classes)

    def forward(self, x, hidden):
        """
        The forward method of the model.
        :param x: Input image
        :param hidden: Hidden input
        :return:
        """
        # this needs to ensure consistency between GPU/CPU training
        hidden = hidden.transpose(0, 1).contiguous()
        # encoding
        x_out_layer1, hidden_out_layer1 = self.gru_encoder(x, hidden)
        # decoding
        x_out_final, hidden_final = self.gru_decoder(x_out_layer1, hidden_out_layer1)

        # classification
        base_out = self.dense1_base(x_out_final)
        rle_out = self.dense2_rle(x_out_final)

        hidden_final = hidden_final.transpose(0, 1).contiguous()
        return base_out, rle_out, hidden_final

    def init_hidden(self, batch_size, num_layers, bidirectional=True):
        """
        Initialize the hidden tensor.
        :param batch_size: Size of the batch
        :param num_layers: Number of GRU layers
        :param bidirectional: If true then GRU layers are bidirectional
        :return:
        """
        num_directions = 1
        if bidirectional:
            num_directions = 2

        return torch.zeros(batch_size, num_directions * num_layers, self.hidden_size)
