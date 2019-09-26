import torch
import os
from modules.python.models.TransducerModel import TransducerGRUBase, TransducerGRURLE


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
    def get_new_gru_model():
        """
        Create a new model object and return it.
        :return: A new model object
        """
        # get a new model
        transducer_model_base = TransducerGRUBase()
        transducer_model_rle = TransducerGRURLE()
        return transducer_model_base, transducer_model_rle

    @staticmethod
    def load_simple_model(model_path):
        """
        This method loads a model from a given model path.
        :param model_path: Path to a model
        :return: A loaded model with some other auxiliary information
        """
        # first load the model to cpu, it's usually a dicttionary
        checkpoint = torch.load(model_path, map_location='cpu')
        # extract auxiliary information from the model dictionary
        epochs = checkpoint['epochs']

        # create a new model
        transducer_model_base, transducer_model_rle = ModelHandler.get_new_gru_model()
        # load the model state/weights
        model_state_dict_base = checkpoint['model_state_dict_base']
        model_state_dict_rle = checkpoint['model_state_dict_rle']

        # now create a state dict that we can load to the new model
        from collections import OrderedDict
        new_model_state_dict_base = OrderedDict()
        new_model_state_dict_rle = OrderedDict()

        for k, v in model_state_dict_base.items():
            name = k
            # this happens due to training on the GPU. It's a pytorch issue, we can't fix it.
            if k[0:7] == 'module.':
                name = k[7:]  # remove `module.`
            new_model_state_dict_base[name] = v

        for k, v in model_state_dict_rle.items():
            name = k
            # this happens due to training on the GPU. It's a pytorch issue, we can't fix it.
            if k[0:7] == 'module.':
                name = k[7:]  # remove `module.`
            new_model_state_dict_rle[name] = v

        # transfer the weights to the new model
        transducer_model_base.load_state_dict(new_model_state_dict_base)
        transducer_model_base.cpu()

        transducer_model_rle.load_state_dict(new_model_state_dict_rle)
        transducer_model_base.cpu()

        # return the loaded model
        return transducer_model_base, transducer_model_rle, epochs

    @staticmethod
    def load_simple_optimizer(transducer_optimizer_base, transducer_optimizer_rle, checkpoint_path, gpu_mode):
        """
        Load the optimizer state. This is required during re-training/transfer learning a previous model.
        :param transducer_optimizer_base: Optimizer state for base
        :param transducer_optimizer_rle: Optimizer state for rle
        :param checkpoint_path: Path to the model
        :param gpu_mode: If True, the model and optimizer will be loaded on CUDA/GPU.
        :return: A loaded optimizer
        """
        if gpu_mode:
            # if gpu is true, then load the optimizer and transfer the state to cuda
            checkpoint = torch.load(checkpoint_path)
            transducer_optimizer_base.load_state_dict(checkpoint['model_optimizer_base'])
            transducer_optimizer_rle.load_state_dict(checkpoint['model_optimizer_rle'])
            for state in transducer_optimizer_base.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

            for state in transducer_optimizer_rle.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        else:
            # if gpu not true, then simply load the optimizer.
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            transducer_optimizer_base.load_state_dict(checkpoint['model_optimizer_base'])
            transducer_optimizer_rle.load_state_dict(checkpoint['model_optimizer_rle'])

        return transducer_optimizer_base, transducer_optimizer_rle

    @staticmethod
    def save_model(transducer_model_base,
                   transducer_model_rle,
                   model_optimizer_base,
                   model_optimizer_rle,
                   epoch,
                   file_name):
        """
        Save the model each epoch.
        :param transducer_model_base: Model for saving
        :param transducer_model_rle: Model for saving
        :param model_optimizer_base: Optimizer used for training the base inference model
        :param model_optimizer_rle: Optimizer used for training the rle inference model
        :param epoch: Number of epochs model has been trained on.
        :param file_name: Name of the file where the MODEL will be saved.
        :return:
        """
        if os.path.isfile(file_name):
            os.remove(file_name)
        ModelHandler.save_checkpoint({
            'model_state_dict_base': transducer_model_base.state_dict(),
            'model_state_dict_rle': transducer_model_rle.state_dict(),
            'model_optimizer_base': model_optimizer_base.state_dict(),
            'model_optimizer_rle': model_optimizer_rle.state_dict(),
            'epochs': epoch,
        }, file_name)


