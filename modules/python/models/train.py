import sys
import torch
import torch.nn as nn
import os
from tqdm import tqdm

from torch.utils.data import DataLoader
from modules.python.models.dataloader import SequenceDataset
from modules.python.TextColor import TextColor
from modules.python.models.test import test
from modules.python.Options import ImageSizeOptions, TrainOptions
from modules.python.models.ModelHander import ModelHandler
"""
This script implements the train method of HELEN.
The train method trains a model on a given set of images and saves model files after each epoch.
"""


def train(train_file,
          test_file,
          batch_size,
          epoch_limit,
          gpu_mode,
          num_workers,
          retrain_model,
          retrain_model_path,
          lr,
          decay,
          model_dir,
          stats_dir,
          not_hyperband):
    """
    This method implements the training scheme of HELEN. It takes a set of training images and a set of testing
    images and trains the model on the training images and after each epoch it evaluates the trained model on the
    test image set. It saves all model state after each epoch regardless of it's performance. It also saves some
    statistics of the saved models and the training run.
    :param train_file: Path to train image set.
    :param test_file: Path to test image set.
    :param batch_size: Batch size for minibatch operation
    :param epoch_limit: Number of iterations the training will go for
    :param gpu_mode: If True, training and testing will be done on GPU
    :param num_workers: Number of workers for dataloader
    :param retrain_model: If True then it will load a previously-trained model for retraining
    :param retrain_model_path: Path to a previously trained model
    :param gru_layers: Number of GRU layers in the model
    :param hidden_size: Hidden size of the model
    :param lr: Learning Rate for the optimizer
    :param decay: Weight Decay for the optimizer
    :param model_dir: Directory where models will be saved
    :param stats_dir: Directory where statistics of this run will be saved
    :param not_hyperband: This is used by hyperband. If True then hyperband is not running.
    :return:
    """

    # if hyperband is not running then create stat logger for train, test and confusion
    if not_hyperband is True:
        train_loss_logger = open(stats_dir + "train_loss.csv", 'w')
        test_loss_logger = open(stats_dir + "test_loss.csv", 'w')
        confusion_matrix_logger = open(stats_dir + "confusion_matrix.txt", 'w')
    else:
        train_loss_logger = None
        test_loss_logger = None
        confusion_matrix_logger = None

    sys.stderr.write(TextColor.PURPLE + 'Loading data\n' + TextColor.END)
    # initialize training dataset loader
    train_data_set = SequenceDataset(train_file)
    train_loader = DataLoader(train_data_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=gpu_mode)

    # if retrain model is true then load the model from the model path
    if retrain_model is True:
        if os.path.isfile(retrain_model_path) is False:
            sys.stderr.write(TextColor.RED + "ERROR: INVALID PATH TO RETRAIN PATH MODEL --retrain_model_path\n")
            exit(1)
        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADING\n" + TextColor.END)
        transducer_model_base, transducer_model_rle, prev_ite = ModelHandler.load_simple_model(retrain_model_path)

        if not_hyperband is True:
            epoch_limit = prev_ite + epoch_limit

        sys.stderr.write(TextColor.GREEN + "INFO: RETRAIN MODEL LOADED\n" + TextColor.END)
    else:
        # if training from scractch, then create a new model
        transducer_model_base, transducer_model_rle = ModelHandler.get_new_gru_model()
        prev_ite = 0

    # count the number of trainable parameters for reporting
    param_count_base = sum(p.numel() for p in transducer_model_base.parameters() if p.requires_grad)
    sys.stderr.write(TextColor.RED + "INFO: TOTAL PARAMETERS FOR BASE INFERENCE:\t" + str(param_count_base) + "\n" + TextColor.END)
    param_count_rle = sum(p.numel() for p in transducer_model_rle.parameters() if p.requires_grad)
    sys.stderr.write(TextColor.RED + "INFO: TOTAL PARAMETERS FOR RLE INFERENCE:\t" + str(param_count_rle) + "\n" + TextColor.END)

    # create a model optimizer
    model_optimizer_base = torch.optim.Adam(transducer_model_base.parameters(), lr=lr, weight_decay=decay)
    model_optimizer_rle = torch.optim.Adam(transducer_model_rle.parameters(), lr=lr, weight_decay=decay)
    # this learning rate scheduler reduces learning rate when model reaches plateau
    lr_scheduler_base = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer_base, 'min')
    lr_scheduler_rle = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer_rle, 'min')

    # if retrain model is true then load the optimizer
    if retrain_model is True:
        sys.stderr.write(TextColor.GREEN + "INFO: OPTIMIZER LOADING\n" + TextColor.END)
        model_optimizer_base, model_optimizer_rle = ModelHandler.load_simple_optimizer(model_optimizer_base,
                                                                                       model_optimizer_rle,
                                                                                       retrain_model_path, gpu_mode)
        sys.stderr.write(TextColor.GREEN + "INFO: OPTIMIZER LOADED\n" + TextColor.END)

    # class_weights = torch.Tensor(TrainOptions.CLASS_WEIGHTS)
    # we perform a multi-task classification, so we need two loss functions, each performing a single task
    # criterion base is the loss function for base prediction
    criterion_base = nn.CrossEntropyLoss()
    # criterion rle is the loss function for RLE prediction
    criterion_rle = nn.CrossEntropyLoss()

    # if gpu mode is true then transfer the model and loss functions to cuda
    if gpu_mode is True:
        transducer_model_base = torch.nn.DataParallel(transducer_model_base).cuda()
        transducer_model_rle = torch.nn.DataParallel(transducer_model_rle).cuda()
        criterion_base = criterion_base.cuda()
        criterion_rle = criterion_rle.cuda()

    start_epoch = prev_ite

    # Train the Model
    sys.stderr.write(TextColor.PURPLE + 'Training starting\n' + TextColor.END)
    # create stats dicts
    stats = dict()
    stats['loss_epoch'] = []
    stats['accuracy_epoch'] = []
    sys.stderr.write(TextColor.BLUE + 'Start: ' + str(start_epoch + 1) + ' End: ' + str(epoch_limit) + "\n")

    # for each epoch we iterate over the training dataset once
    for epoch in range(start_epoch, epoch_limit, 1):
        total_loss_base = 0
        total_loss_rle = 0
        total_loss = 0
        total_images = 0
        sys.stderr.write(TextColor.BLUE + 'Train epoch: ' + str(epoch + 1) + "\n")
        batch_no = 1

        # tqdm is the progress bar we use for logging
        with tqdm(total=len(train_loader), desc='Loss', leave=True, ncols=100) as progress_bar:
            # make sure the model is in train mode. BN is different in train and eval.
            transducer_model_base.train()
            transducer_model_rle.train()

            for base_channel, rle_channels, normalization, label_base, label_rle in train_loader:
                # convert the tensors to the proper datatypes.
                base_image = base_channel.type(torch.FloatTensor)
                rle_image = rle_channels.type(torch.FloatTensor)
                label_base = label_base.type(torch.LongTensor)
                label_rle = label_rle.type(torch.LongTensor)

                # initialize the hidden input for the first chunk
                hidden = torch.zeros(base_image.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)
                hidden_rle_a = torch.zeros(rle_image.size(0), 2 * TrainOptions.RLE_GRU_LAYERS, TrainOptions.RLE_HIDDEN_SIZE)
                hidden_rle_c = torch.zeros(rle_image.size(0), 2 * TrainOptions.RLE_GRU_LAYERS, TrainOptions.RLE_HIDDEN_SIZE)
                hidden_rle_g = torch.zeros(rle_image.size(0), 2 * TrainOptions.RLE_GRU_LAYERS, TrainOptions.RLE_HIDDEN_SIZE)
                hidden_rle_t = torch.zeros(rle_image.size(0), 2 * TrainOptions.RLE_GRU_LAYERS, TrainOptions.RLE_HIDDEN_SIZE)
                hidden_rle_combined = torch.zeros(rle_image.size(0), 2 * TrainOptions.RLE_GRU_LAYERS, TrainOptions.RLE_HIDDEN_SIZE)

                # if gpu_mode is true then transfer all tensors to cuda
                if gpu_mode:
                    base_image = base_image.cuda()
                    rle_image = rle_image.cuda()
                    label_base = label_base.cuda()
                    label_rle = label_rle.cuda()
                    hidden = hidden.cuda()
                    hidden_rle_a = hidden_rle_a.cuda()
                    hidden_rle_c = hidden_rle_c.cuda()
                    hidden_rle_g = hidden_rle_g.cuda()
                    hidden_rle_t = hidden_rle_t.cuda()
                    hidden_rle_combined = hidden_rle_combined.cuda()

                # perform a sliding window on the entire image sequence length
                for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                    # we optimize over each chunk
                    model_optimizer_base.zero_grad()
                    model_optimizer_rle.zero_grad()

                    # if current position + window size goes beyond the size of the window,
                    # that means we've reached the end
                    if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                        break

                    # get the chunks for this window
                    base_image_chunk = base_image[:, i:i+TrainOptions.TRAIN_WINDOW]
                    rle_image_chunk = rle_image[:, :, i:i+TrainOptions.TRAIN_WINDOW]
                    label_base_chunk = label_base[:, i:i+TrainOptions.TRAIN_WINDOW]
                    label_rle_chunk = label_rle[:, i:i+TrainOptions.TRAIN_WINDOW]

                    # get the base inference from the model
                    base_out, hidden = transducer_model_base(base_image_chunk, hidden)
                    # calculate loss for base prediction
                    loss_base = criterion_base(base_out.contiguous().view(-1, TrainOptions.TOTAL_BASE_LABELS),
                                               label_base_chunk.contiguous().view(-1))
                    # backpropagation and weight update
                    loss_base.backward()
                    model_optimizer_base.step()

                    hidden = hidden.detach()
                    base_out = base_out.detach()

                    rle_out, hidden_rle_a, hidden_rle_c, hidden_rle_g, hidden_rle_t, hidden_rle_combined = \
                        transducer_model_rle(rle_image_chunk, base_out, hidden_rle_combined, hidden_rle_a,
                                             hidden_rle_c, hidden_rle_g, hidden_rle_t)

                    # calculate loss for RLE prediction
                    loss_rle = criterion_rle(rle_out.contiguous().view(-1, TrainOptions.TOTAL_RLE_LABELS),
                                             label_rle_chunk.contiguous().view(-1))

                    loss_rle.backward()
                    model_optimizer_rle.step()

                    # update the loss values
                    total_loss_base += loss_base.item()
                    total_loss_rle += loss_rle.item()
                    total_images += base_image.size(0)

                    # detach the hidden from the graph as the next chunk will be a new optimization
                    hidden_rle_a = hidden_rle_a.detach()
                    hidden_rle_c = hidden_rle_c.detach()
                    hidden_rle_g = hidden_rle_g.detach()
                    hidden_rle_t = hidden_rle_t.detach()
                    hidden_rle_combined = hidden_rle_combined.detach()

                # update the progress bar
                avg_loss = (total_loss / total_images) if total_images else 0
                progress_bar.set_description("Base: " + str(round(total_loss_base, 4)) +
                                             ", RLE: " + str(round(total_loss_rle, 4)))

                if not_hyperband is True:
                    train_loss_logger.write(str(epoch + 1) + "," + str(batch_no) + "," + str(avg_loss) + "\n")
                progress_bar.refresh()
                progress_bar.update(1)
                batch_no += 1

            progress_bar.close()

        # after each epoch, evaluate the current state of the model
        stats_dictionary = test(test_file, batch_size, gpu_mode, transducer_model_base, transducer_model_rle,
                                num_workers)

        lr_scheduler_base.step(stats_dictionary['loss_base'])
        lr_scheduler_rle.step(stats_dictionary['loss_rle'])

        # save the model after each epoch and update the loggers after each epoch
        if not_hyperband is True:
            ModelHandler.save_model(transducer_model_base,
                                    transducer_model_rle,
                                    model_optimizer_base,
                                    model_optimizer_rle,
                                    epoch, model_dir + "HELEN_epoch_" + str(epoch + 1) + '_checkpoint.pkl')
            sys.stderr.write(TextColor.RED + "\nMODEL SAVED SUCCESSFULLY.\n" + TextColor.END)

            test_loss_logger.write(str(epoch + 1) + "," + str(stats_dictionary['loss_base']) + ","
                                   + str(stats_dictionary['loss_rle']) + ","
                                   + str(stats_dictionary['accuracy_base']) + ","
                                   + str(stats_dictionary['accuracy_rle']) + "\n")

            confusion_matrix_logger.write(str(epoch + 1) + "\n" + str(stats_dictionary['base_confusion_matrix']) + "\n")
            train_loss_logger.flush()
            test_loss_logger.flush()
            confusion_matrix_logger.flush()
        # else:
        #     # this setup is for hyperband
        #     if epoch + 1 >= 2 and stats['accuracy'] < 90:
        #         sys.stderr.write(TextColor.PURPLE + 'EARLY STOPPING AS THE MODEL NOT DOING WELL\n' + TextColor.END)
        #         return transducer_model, model_optimizer, stats

    # notify that the model has finished training.
    sys.stderr.write(TextColor.PURPLE + 'Finished training\n' + TextColor.END)
