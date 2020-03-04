import torch
import os
from helen.modules.python.models.TransducerModel import TransducerGRU


class ModelHandler:
    """
    The ModelHandler class handles different model saving/loading operations that we do.
    """
    @staticmethod
    def save_checkpoint(state, filename):
        """
        Save a model checkpoint to a file.
        :param state: A model state (usually the model itself in a dictionary format)
        :param filename: Name of the output file
        :return:
        """
        torch.save(state, filename)

    @staticmethod
    def get_new_gru_model(input_channels, image_features, gru_layers, hidden_size, num_base_classes, num_rle_classes):
        """
        Create a new model object and return it.
        :param input_channels: Number of channels in the input image (usually 1)
        :param image_features: Number of features in one column of the pileup
        :param gru_layers: Number of layers in the transducer model
        :param hidden_size: The size of the hidden layer
        :param num_base_classes: Number of base classes
        :param num_rle_classes: Number of RLE classes
        :return: A new model object
        """
        # get a new model
        transducer_model = TransducerGRU(input_channels, image_features, gru_layers, hidden_size, num_base_classes,
                                         num_rle_classes, bidirectional=True)
        return transducer_model

    @staticmethod
    def load_simple_model(model_path, input_channels, image_features, seq_len, num_base_classes, num_rle_classes):
        """
        This method loads a model from a given model path.
        :param model_path: Path to a model
        :param input_channels: Number of channels in the input image (usually 1)
        :param image_features: Number of features in one column of the pileup
        :param seq_len: Length of the sequence in one image
        :param num_base_classes: Number of base classes
        :param num_rle_classes: Number of RLE classes
        :return: A loaded model with some other auxiliary information
        """
        # first load the model to cpu, it's usually a dicttionary
        checkpoint = torch.load(model_path, map_location='cpu')
        # extract auxiliary information from the model dictionary
        hidden_size = checkpoint['hidden_size']
        gru_layers = checkpoint['gru_layers']
        epochs = checkpoint['epochs']

        # create a new model
        transducer_model = ModelHandler.get_new_gru_model(input_channels=input_channels,
                                                          image_features=image_features,
                                                          gru_layers=gru_layers,
                                                          hidden_size=hidden_size,
                                                          num_base_classes=num_base_classes,
                                                          num_rle_classes=num_rle_classes)
        # load the model state/weights
        model_state_dict = checkpoint['model_state_dict']

        # now create a state dict that we can load to the new model
        from collections import OrderedDict
        new_model_state_dict = OrderedDict()

        for k, v in model_state_dict.items():
            name = k
            # this happens due to training on the GPU. It's a pytorch issue, we can't fix it.
            if k[0:7] == 'module.':
                name = k[7:]  # remove `module.`
            new_model_state_dict[name] = v

        # transfer the weights to the new model
        transducer_model.load_state_dict(new_model_state_dict)
        transducer_model.cpu()

        # return the loaded model
        return transducer_model, hidden_size, gru_layers, epochs

    @staticmethod
    def load_simple_optimizer(transducer_optimizer, checkpoint_path, gpu_mode):
        """
        Load the optimizer state. This is required during re-training/transfer learning a previous model.
        :param transducer_optimizer: Optimizer state
        :param checkpoint_path: Path to the model
        :param gpu_mode: If True, the model and optimizer will be loaded on CUDA/GPU.
        :return: A loaded optimizer
        """
        if gpu_mode:
            # if gpu is true, then load the optimizer and transfer the state to cuda
            checkpoint = torch.load(checkpoint_path)
            transducer_optimizer.load_state_dict(checkpoint['model_optimizer'])
            for state in transducer_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            # if gpu not true, then simply load the optimizer.
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            transducer_optimizer.load_state_dict(checkpoint['model_optimizer'])

        return transducer_optimizer

    @staticmethod
    def save_model(transducer_model,
                   model_optimizer,
                   hidden_size,
                   layers,
                   epoch,
                   file_name):
        """
        Save the model each epoch.
        :param transducer_model: Model for saving
        :param model_optimizer: Optimizer used for training the model
        :param hidden_size: Hidden layer size of the model
        :param layers: Number of layers in the model
        :param epoch: Number of epochs model has been trained on.
        :param file_name: Name of the file where the MODEL will be saved.
        :return:
        """
        if os.path.isfile(file_name):
            os.remove(file_name)
        ModelHandler.save_checkpoint({
            'model_state_dict': transducer_model.state_dict(),
            'model_optimizer': model_optimizer.state_dict(),
            'hidden_size': hidden_size,
            'gru_layers': layers,
            'epochs': epoch,
        }, file_name)


