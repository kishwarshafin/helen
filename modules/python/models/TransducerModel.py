import torch
import torch.nn as nn
import warnings
from modules.python.Options import TrainOptions, ImageSizeOptions

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
    def __init__(self):
        """
        The initialization of the model
        """
        super(TransducerGRU, self).__init__()

        # the gru encoder-decoder layer for base
        self.gru_encoder = nn.GRU(ImageSizeOptions.BASE_IMAGE_HEIGHT, TrainOptions.HIDDEN_SIZE,
                                  num_layers=TrainOptions.GRU_LAYERS,
                                  bidirectional=True,
                                  batch_first=True)

        self.gru_decoder = nn.GRU(2 * TrainOptions.HIDDEN_SIZE, TrainOptions.HIDDEN_SIZE,
                                  num_layers=TrainOptions.GRU_LAYERS,
                                  bidirectional=True,
                                  batch_first=True)

        # the gru encoder-decoder layer for RLE on A base
        self.rle_encoder_A = nn.GRU(ImageSizeOptions.RLE_IMAGE_HEIGHT, TrainOptions.RLE_HIDDEN_SIZE,
                                    num_layers=TrainOptions.RLE_GRU_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)

        self.rle_decoder_A = nn.GRU(2 * TrainOptions.RLE_HIDDEN_SIZE, TrainOptions.RLE_HIDDEN_SIZE,
                                    num_layers=TrainOptions.RLE_GRU_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)

        # the gru encoder-decoder layer for RLE on C base
        self.rle_encoder_C = nn.GRU(ImageSizeOptions.RLE_IMAGE_HEIGHT, TrainOptions.RLE_HIDDEN_SIZE,
                                    num_layers=TrainOptions.RLE_GRU_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)

        self.rle_decoder_C = nn.GRU(2 * TrainOptions.RLE_HIDDEN_SIZE, TrainOptions.RLE_HIDDEN_SIZE,
                                    num_layers=TrainOptions.RLE_GRU_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)

        # the gru encoder-decoder layer for RLE on G base
        self.rle_encoder_G = nn.GRU(ImageSizeOptions.RLE_IMAGE_HEIGHT, TrainOptions.RLE_HIDDEN_SIZE,
                                    num_layers=TrainOptions.RLE_GRU_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)

        self.rle_decoder_G = nn.GRU(2 * TrainOptions.RLE_HIDDEN_SIZE, TrainOptions.RLE_HIDDEN_SIZE,
                                    num_layers=TrainOptions.RLE_GRU_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)

        # the gru encoder-decoder layer for RLE on T base
        self.rle_encoder_T = nn.GRU(ImageSizeOptions.RLE_IMAGE_HEIGHT, TrainOptions.RLE_HIDDEN_SIZE,
                                    num_layers=TrainOptions.RLE_GRU_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)

        self.rle_decoder_T = nn.GRU(2 * TrainOptions.RLE_HIDDEN_SIZE, TrainOptions.RLE_HIDDEN_SIZE,
                                    num_layers=TrainOptions.RLE_GRU_LAYERS,
                                    bidirectional=True,
                                    batch_first=True)

        # the linear layer for base and RLE classification
        self.dense1_base = nn.Linear(TrainOptions.HIDDEN_SIZE * 2, TrainOptions.TOTAL_BASE_LABELS)
        self.base_softmax = nn.Softmax(dim=2)

        self.dense_rleA = nn.Linear(TrainOptions.RLE_HIDDEN_SIZE * 2, TrainOptions.TOTAL_RLE_LABELS)
        self.dense_rleC = nn.Linear(TrainOptions.RLE_HIDDEN_SIZE * 2, TrainOptions.TOTAL_RLE_LABELS)
        self.dense_rleG = nn.Linear(TrainOptions.RLE_HIDDEN_SIZE * 2, TrainOptions.TOTAL_RLE_LABELS)
        self.dense_rleT = nn.Linear(TrainOptions.RLE_HIDDEN_SIZE * 2, TrainOptions.TOTAL_RLE_LABELS)
        self.rle_a_softmax = nn.Softmax(dim=2)
        self.rle_c_softmax = nn.Softmax(dim=2)
        self.rle_g_softmax = nn.Softmax(dim=2)
        self.rle_t_softmax = nn.Softmax(dim=2)

        self.total_rle_features = TrainOptions.TOTAL_BASE_LABELS + 4 * TrainOptions.TOTAL_RLE_LABELS
        self.dense_rle_layer1 = nn.Linear(self.total_rle_features, 2 * self.total_rle_features)
        self.dense_rle_layer2 = nn.Linear(2 * self.total_rle_features, 4 * self.total_rle_features)
        self.dense_rle_layer3 = nn.Linear(4 * self.total_rle_features, TrainOptions.TOTAL_RLE_LABELS)
        self.rle_prob = nn.Softmax(dim=2)

    def forward(self, x_base, x_rle, hidden, hidden_rle_a, hidden_rle_c, hidden_rle_g, hidden_rle_t):
        """
        The forward method of the model.
        :param x_base: Input base image
        :param x_rle: Input RLE image
        :param hidden: Hidden input
        :return:
        """
        # this needs to ensure consistency between GPU/CPU training
        hidden = hidden.transpose(0, 1).contiguous()
        hidden_rle_a = hidden_rle_a.transpose(0, 1).contiguous()
        hidden_rle_c = hidden_rle_c.transpose(0, 1).contiguous()
        hidden_rle_g = hidden_rle_g.transpose(0, 1).contiguous()
        hidden_rle_t = hidden_rle_t.transpose(0, 1).contiguous()

        # encoding
        x_out_layer1, hidden_out_layer1 = self.gru_encoder(x_base, hidden)
        # decoding
        x_out_final, hidden_final = self.gru_decoder(x_out_layer1, hidden_out_layer1)

        # classification
        base_out = self.dense1_base(x_out_final)

        base_prob_layer = nn.Softmax(dim=2)
        base_prob = base_prob_layer(base_out)

        rle_a_features = x_rle[:, 0]
        rle_c_features = x_rle[:, 1]
        rle_g_features = x_rle[:, 2]
        rle_t_features = x_rle[:, 3]

        # encoding-decoding RLE A
        x_out_rle_a, hidden_out_rle_a = self.rle_encoder_A(rle_a_features, hidden_rle_a)
        x_out_rle_a_final, hidden_rle_a_final = self.rle_decoder_A(x_out_rle_a, hidden_out_rle_a)
        x_out_rle_a_out = self.dense_rleA(x_out_rle_a_final)
        rle_a_prob = self.rle_a_softmax(x_out_rle_a_out)

        # encoding-decoding RLE C
        x_out_rle_c, hidden_out_rle_c = self.rle_encoder_C(rle_c_features, hidden_rle_c)
        x_out_rle_c_final, hidden_rle_c_final = self.rle_decoder_C(x_out_rle_c, hidden_out_rle_c)
        x_out_rle_c_out = self.dense_rleC(x_out_rle_c_final)
        rle_c_prob = self.rle_c_softmax(x_out_rle_c_out)

        # encoding-decoding RLE G
        x_out_rle_g, hidden_out_rle_g = self.rle_encoder_G(rle_g_features, hidden_rle_g)
        x_out_rle_g_final, hidden_rle_g_final = self.rle_decoder_G(x_out_rle_g, hidden_out_rle_g)
        x_out_rle_g_out = self.dense_rleG(x_out_rle_g_final)
        rle_g_prob = self.rle_g_softmax(x_out_rle_g_out)

        # encoding-decoding RLE T
        x_out_rle_t, hidden_out_rle_t = self.rle_encoder_T(rle_t_features, hidden_rle_t)
        x_out_rle_t_final, hidden_rle_t_final = self.rle_decoder_T(x_out_rle_t, hidden_out_rle_t)
        x_out_rle_t_out = self.dense_rleT(x_out_rle_t_final)
        rle_t_prob = self.rle_t_softmax(x_out_rle_t_out)

        all_rle_features = torch.cat([base_prob, rle_a_prob, rle_c_prob, rle_g_prob, rle_t_prob],
                                     dim=2)

        rle_out_layer1 = self.dense_rle_layer1(all_rle_features)
        rle_out_layer2 = self.dense_rle_layer2(rle_out_layer1)
        rle_out = self.dense_rle_layer3(rle_out_layer2)
        rle_prob = self.rle_prob(rle_out)

        hidden_final = hidden_final.transpose(0, 1).contiguous()
        hidden_rle_a_final = hidden_rle_a_final.transpose(0, 1).contiguous()
        hidden_rle_c_final = hidden_rle_c_final.transpose(0, 1).contiguous()
        hidden_rle_g_final = hidden_rle_g_final.transpose(0, 1).contiguous()
        hidden_rle_t_final = hidden_rle_t_final.transpose(0, 1).contiguous()

        return base_out, base_prob, rle_out, rle_prob, hidden_final, hidden_rle_a_final, hidden_rle_c_final, \
            hidden_rle_g_final, hidden_rle_t_final

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
