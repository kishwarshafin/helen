import sys
import torch
from tqdm import tqdm
import torchnet.meter as meter
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from helen.modules.python.models.dataloader import SequenceDataset
from helen.modules.python.TextColor import TextColor
from helen.modules.python.Options import ImageSizeOptions, TrainOptions
"""
This script implements the test method.
The test method evaluates a trained model on a given test dataset and report various accuracy matrices.
"""


def test(data_filepath,
         batch_size,
         gpu_mode,
         transducer_model,
         num_workers,
         gru_layers,
         hidden_size,
         num_base_classes,
         num_rle_classes,
         print_details=False):
    """
    This method performs testing of a trained model.
    :param data_filepath: Path to a directory containing all labeled h5 files used for testing
    :param batch_size: Size of minibatch
    :param gpu_mode: If true, the model will be loaded on GPU using CUDA.
    :param transducer_model: A trained model
    :param num_workers: Number of thread workers for dataloader
    :param gru_layers: Number of GRU layers
    :param hidden_size: Hidden size of the model.
    :param num_base_classes: Number of classes for base prediction
    :param num_rle_classes: Number of classes for RLE prediction
    :param print_details: A debug parameter.
    :return:
    """
    # data loader
    test_data = SequenceDataset(data_filepath)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=gpu_mode)

    # set the model to evaluation mode
    transducer_model.eval()

    # create a loss function for tracking loss during testing
    class_weights = torch.Tensor(TrainOptions.CLASS_WEIGHTS)
    # Loss not doing class weights for the first pass
    criterion_base = nn.CrossEntropyLoss()
    criterion_rle = nn.CrossEntropyLoss(weight=class_weights)

    # if gpu is true then transfer the loss over cuda
    if gpu_mode is True:
        criterion_base = criterion_base.cuda()
        criterion_rle = criterion_rle.cuda()

    # initialize base and rle confusion matrix
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    base_confusion_matrix = meter.ConfusionMeter(num_base_classes)
    rle_confusion_matrix = meter.ConfusionMeter(num_rle_classes)

    # initialize the accuracy matrices
    total_loss = 0
    total_loss_rle = 0
    total_images = 0
    accuracy = 0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Accuracy: ', leave=True, ncols=100) as pbar:
            # iterate over the dataset in minibatch
            for ii, (images, label_base, label_rle) in enumerate(test_loader):
                # convert the tensors to a proper datatype
                images = images.type(torch.FloatTensor)
                label_base = label_base.type(torch.LongTensor)
                label_rle = label_rle.type(torch.LongTensor)

                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    label_base = label_base.cuda()
                    label_rle = label_rle.cuda()

                # initialize the hidden input for the first chunk
                hidden = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)

                if gpu_mode:
                    hidden = hidden.cuda()

                # slide over the image in a sliding window manner
                for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                    # if current position + window size goes beyond the size of the window,
                    # that means we've reached the end
                    if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                        break

                    # chunk the images and labels to this window
                    image_chunk = images[:, i:i+TrainOptions.TRAIN_WINDOW]
                    label_base_chunk = label_base[:, i:i+TrainOptions.TRAIN_WINDOW]
                    label_rle_chunk = label_rle[:, i:i+TrainOptions.TRAIN_WINDOW]

                    # perform an inference on the images
                    output_base, output_rle, hidden = transducer_model(image_chunk, hidden)

                    # calculate loss between the prediction and the true labels
                    loss_base = criterion_base(output_base.contiguous().view(-1, num_base_classes),
                                               label_base_chunk.contiguous().view(-1))
                    loss_rle = criterion_rle(output_rle.contiguous().view(-1, num_rle_classes),
                                             label_rle_chunk.contiguous().view(-1))

                    loss = loss_base + loss_rle

                    # populate the confusion matrix
                    base_confusion_matrix.add(output_base.data.contiguous().view(-1, num_base_classes),
                                              label_base_chunk.data.contiguous().view(-1))
                    rle_confusion_matrix.add(output_rle.data.contiguous().view(-1, num_rle_classes),
                                             label_rle_chunk.data.contiguous().view(-1))

                    total_loss += loss.item()
                    total_images += images.size(0)
                    total_loss_rle += loss_rle.item()

                pbar.update(1)
                # we calculate the accuracy using the confusion matrix
                base_cm_value = base_confusion_matrix.value()
                rle_cm_value = rle_confusion_matrix.value()
                # the sum of all cells in the confusion matrix is the denominator
                base_denom = base_cm_value.sum()
                rle_denom = rle_cm_value.sum()

                # calculate the correct predictions in each case
                base_corrects = 0
                for label in range(0, ImageSizeOptions.TOTAL_BASE_LABELS):
                    base_corrects = base_corrects + base_cm_value[label][label]

                rle_corrects = 0
                for label in range(0, ImageSizeOptions.TOTAL_RLE_LABELS):
                    rle_corrects = rle_corrects + rle_cm_value[label][label]

                # calculate the accuracy
                base_accuracy = 100.0 * (base_corrects / max(1.0, base_denom))
                rle_accuracy = 100.0 * (rle_corrects / max(1.0, rle_denom))

                # set the tqdm bar's accuracy and loss value
                pbar.set_description("Base acc: " + str(round(base_accuracy, 4)) +
                                     ", RLE acc: " + str(round(rle_accuracy, 4)) +
                                     ", RLE loss: " + str(round(total_loss_rle, 4)))

    avg_loss = total_loss / total_images if total_images else 0
    np.set_printoptions(threshold=np.inf)
    # print some statistics
    sys.stderr.write(TextColor.YELLOW+'\nTest Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write(TextColor.BLUE + "Base Confusion Matrix: \n" + str(base_confusion_matrix.value()) + "\n" + TextColor.END)
    sys.stderr.write(TextColor.RED + "RLE Confusion Matrix: \n" + TextColor.END)
    for row in rle_confusion_matrix.value():
        row = row.tolist()
        for elem in row:
            sys.stderr.write("{:9d} ".format(elem))
        sys.stderr.write("\n")

    return {'loss': avg_loss, 'accuracy': accuracy, 'base_confusion_matrix': base_confusion_matrix.conf,
            'rle_confusion_matrix': rle_confusion_matrix.conf}
