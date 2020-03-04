import sys
import torch
import torchnet.meter as meter
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from helen.modules.python.models.dataloader_debug import SequenceDataset
from helen.modules.python.TextColor import TextColor
from helen.modules.python.Options import ImageSizeOptions, TrainOptions
"""
WARNING: THIS IS A DEBUGGING TOOL INTENDED TO BE USED BY THE DEVELOPERS ONLY.
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


label_decoder = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 0: '-'}


def test(data_file, batch_size, gpu_mode, transducer_model, num_workers, gru_layers, hidden_size, output_directory,
         num_base_classes, num_rle_classes, print_details=False):
    # transformations = transforms.Compose([transforms.ToTensor()])
    # torch.set_num_threads(num_workers)
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

    test_file_logger = None
    if print_details:
        test_file_logger = open(output_directory + "prediction_debug.txt", 'w')
        sys.stdout = test_file_logger

    # class_weights = torch.Tensor(CLASS_WEIGHTS)
    # Loss not doing class weights for the first pass
    criterion_base = nn.CrossEntropyLoss()
    criterion_rle = nn.CrossEntropyLoss()

    if gpu_mode is True:
        criterion_base = criterion_base.cuda()
        criterion_rle = criterion_rle.cuda()

    # Test the Model
    sys.stderr.write(TextColor.PURPLE + 'Test starting\n' + TextColor.END)
    base_confusion_matrix = meter.ConfusionMeter(num_base_classes)
    rle_confusion_matrix = meter.ConfusionMeter(num_rle_classes)

    total_loss = 0
    total_loss_rle = 0
    total_images = 0
    accuracy = 0
    prediction_name_set = set()
    with torch.no_grad():
        pbar = tqdm(total=len(test_loader), desc='Accuracy: ', leave=True, ncols=100)
        for ii, (images, label_base, label_run_length, position, contig, contig_start, contig_end, chunk_id, filename) in \
                enumerate(test_loader):
            images = images.type(torch.FloatTensor)
            label_base = label_base.type(torch.LongTensor)
            label_rle = label_run_length.type(torch.LongTensor)
            if gpu_mode:
                # encoder_hidden = encoder_hidden.cuda()
                images = images.cuda()
                label_base = label_base.cuda()
                label_rle = label_rle.cuda()
            hidden = torch.zeros(images.size(0), 2 * TrainOptions.GRU_LAYERS, TrainOptions.HIDDEN_SIZE)
            if gpu_mode:
                hidden = hidden.cuda()
            for i in range(0, ImageSizeOptions.SEQ_LENGTH, TrainOptions.WINDOW_JUMP):
                if i + TrainOptions.TRAIN_WINDOW > ImageSizeOptions.SEQ_LENGTH:
                    break
                image_chunk = images[:, i:i+TrainOptions.TRAIN_WINDOW]
                label_base_chunk = label_base[:, i:i+TrainOptions.TRAIN_WINDOW]
                label_rle_chunk = label_rle[:, i:i+TrainOptions.TRAIN_WINDOW]
                position_chunk = position[:, i:i+TrainOptions.TRAIN_WINDOW]
                output_base, output_rle, hidden = transducer_model(image_chunk, hidden)

                # do softmax and get prediction
                m = nn.Softmax(dim=2)
                soft_probs = m(output_base)
                output_preds = soft_probs.cpu()
                base_max_value, predicted_base_label = torch.max(output_preds, dim=2)

                # convert everything to list
                predicted_base_label = predicted_base_label.numpy().tolist()

                # do softmax and get prediction for rle
                m_rle = nn.Softmax(dim=2)
                rle_soft_probs = m_rle(output_rle)
                rle_output_preds = rle_soft_probs.cpu()
                rle_max_value, predicted_rle_labels = torch.max(rle_output_preds, dim=2)
                predicted_rle_labels = predicted_rle_labels.data.numpy().tolist()

                true_bases = label_base_chunk.cpu().contiguous().numpy().tolist()
                true_rles = label_rle_chunk.cpu().numpy().tolist()

                for i in range(image_chunk.size(0)):
                    column_count = 0
                    for pred_rle, true_rle, pred_base, true_base, pos in zip(predicted_rle_labels[i], true_rles[i],
                                                                             predicted_base_label[i], true_bases[i],
                                                                             position_chunk[i]):
                        if pred_rle != true_rle[0]:
                            prediction_name = str(contig[i]) + "_" + str(contig_start[i].item()) + "_" \
                                              + str(contig_end[i].item()) + "_" + str(chunk_id[i].item()) + "_" \
                                              + str(pos[0].item()) + "_" + str(pos[1].item()) + "_" + str(pos[2].item())
                            if prediction_name in prediction_name_set:
                                column_count += 1
                                continue
                            prediction_name_set.add(prediction_name)

                            if print_details:
                                print("RLE TRUE/PRED: " + "{:02d}/{:02d}".format(true_rle[0] , pred_rle) + ",",
                                      "BASE TRUE/PRED: " + "{}/{}".format(label_decoder[true_base[0]], label_decoder[pred_base])+ ",",
                                      "CONTIG: " + (str(contig[i]) + ":" + str(contig_start[i].item()) + "-" + str(contig_end[i].item()))+ ",",
                                      "CHUNK ID: " + str(chunk_id[i].item())+ ",",
                                      "POS: {:3d} {:3d} INDEX: {:2d} SPLIT INDEX:{:2d}".format(pos[0].item() + contig_start[i].item(), pos[0].item(), pos[1].item(), pos[2].item())+ ",",
                                      "FILENAME: ", str(filename[i])+ ",")
                                print(''.join(['-'] * 93))
                                print("R: ", ' '.join(["{:5d}".format(int(x/2)) if x % 2 == 0 else ' ' for x in range(0, 21)]), '  |')
                                print(''.join(['-'] * 93))
                                print("A: ", ' '.join(["{:3d}".format(int(x)) for x in image_chunk[i, column_count].numpy().tolist()[0:22]]), '|')
                                print("C: ", ' '.join(["{:3d}".format(int(x)) for x in image_chunk[i, column_count].numpy().tolist()[22:44]]), '|')
                                print("G: ", ' '.join(["{:3d}".format(int(x)) for x in image_chunk[i, column_count].numpy().tolist()[44:66]]), '|')
                                print("T: ", ' '.join(["{:3d}".format(int(x)) for x in image_chunk[i, column_count].numpy().tolist()[66:88]]), '|')
                                print("*: ", ' '.join(["{:3d}".format(int(x)) for x in image_chunk[i, column_count].numpy().tolist()[88:]]), ' '*79, '|')
                                print(''.join(['-'] * 93))

                        column_count += 1

                loss_base = criterion_base(output_base.contiguous().view(-1, num_base_classes),
                                           label_base_chunk.contiguous().view(-1))
                loss_rle = criterion_rle(output_rle.contiguous().view(-1, num_rle_classes),
                                         label_rle_chunk.contiguous().view(-1))
                loss = loss_base + loss_rle
                base_confusion_matrix.add(output_base.data.contiguous().view(-1, num_base_classes),
                                          label_base_chunk.data.contiguous().view(-1))
                rle_confusion_matrix.add(output_rle.data.contiguous().view(-1, num_rle_classes),
                                         label_rle_chunk.data.contiguous().view(-1))
                total_loss += loss.item()
                total_images += images.size(0)
                total_loss_rle += loss_rle.item()

            base_cm_value = base_confusion_matrix.value()
            rle_cm_value = rle_confusion_matrix.value()

            base_denom = base_cm_value.sum()
            # rle_denom = rle_cm_value.sum() - rle_cm_value[0][0]
            rle_denom = rle_cm_value.sum()
            base_corrects = 0

            for label in range(0, ImageSizeOptions.TOTAL_BASE_LABELS):
                base_corrects = base_corrects + base_cm_value[label][label]
            rle_corrects = 0

            for label in range(0, ImageSizeOptions.TOTAL_RLE_LABELS):
                rle_corrects = rle_corrects + rle_cm_value[label][label]
                # rle_denom = rle_denom - rle_cm_value[0][label]

            # calculate the accuracy
            base_accuracy = 100.0 * (base_corrects / max(1.0, base_denom))
            rle_accuracy = 100.0 * (rle_corrects / max(1.0, rle_denom))

            # set the tqdm bar's accuracy and loss value
            pbar.update(1)
            pbar.set_description("Base acc: " + str(round(base_accuracy, 4)) +
                                 ", RLE acc: " + str(round(rle_accuracy, 4)))

    avg_loss = total_loss / total_images if total_images else 0
    np.set_printoptions(threshold=np.inf)
    pbar.close()

    sys.stderr.write(TextColor.YELLOW+'\nTest Loss: ' + str(avg_loss) + "\n"+TextColor.END)
    sys.stderr.write(TextColor.BLUE + "Base Confusion Matrix: \n" + str(base_confusion_matrix.value()) + "\n" + TextColor.END)
    sys.stderr.write(TextColor.RED + "RLE Confusion Matrix: \n" + str(rle_confusion_matrix.value()) + "\n" + TextColor.END)
    # sys.stderr.write("label\t\tprecision\n")
    # for label in range(0, ImageSizeOptions.TOTAL_LABELS):
    #     sys.stderr.write(str(label_to_literal(label)) + '\t' + str(precision(label, confusion_matrix.conf)) + "\n")

    return {'loss': avg_loss, 'accuracy': accuracy, 'base_confusion_matrix': base_confusion_matrix.conf,
            'rle_confusion_matrix': rle_confusion_matrix.conf}
