import sys
import torch
from tqdm import tqdm
import torchnet.meter as meter
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from modules.python.models.dataloader_test import SequenceDataset
from modules.python.TextColor import TextColor
from modules.python.Options import ImageSizeOptions, TrainOptions
"""
This script will evaluate a model and return the loss value.

Input:
- A trained model
- A test CSV file to evaluate

Returns:
- Loss value
"""
# CLASS_WEIGHTS = [0.3, 1.0, 1.0, 1.0, 1.0]


def precision(label, confusion_matrix):
    col = confusion_matrix[:, label]
    return round(confusion_matrix[label, label] / max(1, col.sum()) , 3)


def label_to_literal(label):
    label_decoder = {'A': 0, 'C': 20, 'G': 40, 'T': 60, '_': 0}
    if label == 0:
        return '-', 0
    if label <= 20:
        return 'A', label
    if label <= 40:
        return 'C', label - 20
    if label <= 60:
        return 'G', label - 40

    return 'T', label - 60


def test(data_file, batch_size, gpu_mode, transducer_model, num_workers, gru_layers, hidden_size,
         num_classes=ImageSizeOptions.TOTAL_LABELS, print_details=False):
    # transformations = transforms.Compose([transforms.ToTensor()])

    # data loader
    test_data = SequenceDataset(data_file)
    test_loader = DataLoader(test_data,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=gpu_mode)
    # sys.stderr.write(TextColor.CYAN + 'Test data loaded\n')

    # set the evaluation mode of the model
    transducer_model.eval()

    # class_weights = torch.Tensor(CLASS_WEIGHTS)
    # Loss not doing class weights for the first pass
    criterion = nn.CrossEntropyLoss()

    if gpu_mode is True:
        criterion = criterion.cuda()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    confusion_matrix = meter.ConfusionMeter(num_classes)

    total_loss = 0
    total_images = 0
    accuracy = 0

    with torch.no_grad():
        with tqdm(total=len(test_loader), desc='Accuracy: ', leave=True, ncols=100) as pbar:
            for ii, (images, labels) in enumerate(test_loader):
                if gpu_mode:
                    # encoder_hidden = encoder_hidden.cuda()
                    images = images.cuda()
                    labels = labels.cuda()

                hidden = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)

                if gpu_mode:
                    hidden = hidden.cuda()

                for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                    if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                        break

                    image_chunk = images[:, i:i+TrainOptions.TRAIN_WINDOW]
                    label_chunk = labels[:, i:i+TrainOptions.TRAIN_WINDOW]
                    output_, hidden = transducer_model(image_chunk, hidden)

                    loss = criterion(output_.contiguous().view(-1, num_classes), label_chunk.contiguous().view(-1))

                    confusion_matrix.add(output_.data.contiguous().view(-1, num_classes),
                                         label_chunk.data.contiguous().view(-1))

                    total_loss += loss.item()
                    total_images += images.size(0)

                pbar.update(1)
                cm_value = confusion_matrix.value()
                denom = cm_value.sum()
                corrects = 0
                for label in range(0, ImageSizeOptions.TOTAL_LABELS):
                    corrects = corrects + cm_value[label][label]
                accuracy = 100.0 * (corrects / max(1.0, denom))
                pbar.set_description("Accuracy: " + str(round(accuracy, 5)))

    avg_loss = total_loss / total_images if total_images else 0
    np.set_printoptions(threshold=np.inf)

    sys.stderr.write(TextColor.YELLOW+'\nTest Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    # sys.stderr.write("Confusion Matrix: \n" + str(class_error_meter.value()) + "\n" + TextColor.END)
    sys.stderr.write("label\t\tprecision\n")
    for label in range(0, ImageSizeOptions.TOTAL_LABELS):
        sys.stderr.write(str(label_to_literal(label)) + '\t' + str(precision(label, confusion_matrix.conf)) + "\n")

    return {'loss': avg_loss, 'accuracy': accuracy, 'confusion_matrix': str(confusion_matrix.conf)}
