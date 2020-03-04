import h5py
import sys
import concurrent.futures
import numpy as np
from collections import defaultdict
import operator
from helen.modules.python.TextColor import TextColor
from helen.modules.python.Options import StitchOptions
from helen.modules.python.FileManager import FileManager
from helen.build import HELEN
import re


class Stitch:
    """
    This class performs stitching of sequences predicted by HELEN in chunks.

    The method is as follows:
    1) MarginPolish
     - MarginPolish takes a "Region" which is usually 1000 bases and generates pileup summaries for them.
     - The region often is more than 1000 bases but for mini-batch to work, we make sure all the images are 1000 bases
       so MarginPolish chunks these Regions into images providing each Regional image with a chunk id.
    2) HELEN call consensus
     - Call Consensus generates a prediction for each of the images, but while saving it makes sure all the images
       that belong to the same region, gets saved under the same chunk_prefix, making sure all the sequences can
     - be aggregated easily.
    3) HELEN stitch
     -  The stitch method that loads one contig at a time and grabs all the "regions" with predictions. The regional
        chunks are easily stitched without having to do an overlap as the positions inside a region is consistent.
     - For regions that are adjacent, HELEN uses a local smith-waterman alignment to find an anchor point to
       stitch sequences from two adjacent sequences together.
    """
    @staticmethod
    def get_confident_positions(alignment):
        """
        Given an alignment object, this method returns a position in both sequences that we can use as
        anchor positions. This method looks for a significant overlap between two sequences that we can use
        as an anchor.
        :param alignment: SSW alignment object.
        :return: two positions
        """
        # get the cigar string and replace '=' and 'X' with 'M' as both can represent a base alignment.
        cigar_string = alignment.cigar_string.replace('=', 'M').replace('X', 'M')

        # this regex will split the cigar string into a list of tuples where each item is (cigar_operation, cigar_len)
        cigar_tuples = re.findall(r'(\d+)(\w)', cigar_string)

        # now we would group the 'M's together which we derived from '=' and 'X'.
        grouped_tuples = list()
        prev_len = 0
        prev_op = None
        for cigar_len, cigar_op in cigar_tuples:
            # if there were no previous operations
            if prev_op is None:
                prev_op = cigar_op
                prev_len = int(cigar_len)
            # if the previous operation is the same as current operation
            elif prev_op == cigar_op:
                # simply extend the operation
                prev_len += int(cigar_len)
            else:
                grouped_tuples.append((prev_op, prev_len))
                prev_op = cigar_op
                prev_len = int(cigar_len)

        # the last one that won't get appeneded
        if prev_op is not None:
            grouped_tuples.append((prev_op, prev_len))

        ref_index = alignment.reference_begin
        read_index = 0

        # now look for an anchor position. We define an anchor position to be a position where we've seen atleast
        # 5 bases overlap between two sequences.
        for cigar_op, cigar_len in grouped_tuples:
            if cigar_op == 'M' and cigar_len >= StitchOptions.OVERLAP_THRESHOLD:
                return ref_index, read_index

            if cigar_op == 'S':
                read_index += cigar_len
            elif cigar_op == 'I':
                read_index += cigar_len
            elif cigar_op == 'D':
                ref_index += cigar_len
            elif cigar_op == 'M':
                ref_index += cigar_len
                read_index += cigar_len
            else:
                # this has never happened so far.
                raise ValueError(TextColor.RED + "ERROR: INVALID CIGAR OPERATION ENCOUNTERED WHILTE STITCHING: "
                                 + str(cigar_op) + "\n")

        # if we can't find any anchors return negative values
        return -1, -1

    def alignment_stitch(self, sequence_chunks):
        """
        This is a stitch worker. This method gets a chunk of contiguous sequence that it stitches.
        The method is very simple, it performs an ssw alignment, finds an anchor position and concatenates
        two adjacent sequences that overlap.
        :param sequence_chunks: A list of sequence chunks in (contig, start, end, sequence) format.
        :return:
        """
        # we make sure that the chunks are sorted by the positions
        sequence_chunks = sorted(sequence_chunks, key=lambda element: (element[1], element[2]))
        # pick the first sequence to be the running sequence
        contig, running_start, running_end, running_sequence = sequence_chunks[0]

        # initialize an ssw aligner
        aligner = HELEN.Aligner(StitchOptions.MATCH_PENALTY, StitchOptions.MISMATCH_PENALTY,
                                StitchOptions.GAP_PENALTY, StitchOptions.GAP_EXTEND_PENALTY)
        # and a filter required by ssw align function
        filter = HELEN.Filter()

        # now iterate through all the chunks
        for i in range(1, len(sequence_chunks)):
            # get the current suquence
            _, this_start, this_end, this_sequence = sequence_chunks[i]
            # make sure the current sequence overlaps with the previously processed sequence
            if this_start < running_end:
                # overlap
                overlap_bases = running_end - this_start
                overlap_bases = overlap_bases + int(overlap_bases * StitchOptions.BASE_ERROR_RATE)

                # now we take the last bases from running sequence
                left_running_sequence_chunk = running_sequence[-overlap_bases:]
                # and first bases from the current sequence
                right_current_sequence = this_sequence[:overlap_bases]

                # initialize an alignment object
                alignment = HELEN.Alignment()
                aligner.SetReferenceSequence(left_running_sequence_chunk, len(left_running_sequence_chunk))
                # align current sequence to the previous sequence
                aligner.Align_cpp(right_current_sequence, filter, alignment, 0)

                # check we have an alignment between the sequences
                if alignment.best_score == 0:
                    sys.stderr.write(TextColor.YELLOW + "WARNING: NO ALIGNMENT FOUND: " + str(this_start)
                                     + " " + str(this_end) + "\n" + TextColor.END)
                    # this is a special case, happens when we encounter a region that is empty. In this case what we do
                    # is append 50 Ns to compensate for the overlap regions and then add the next chunk. This happens
                    # very rarely but happens for sure.
                    if len(right_current_sequence) > 10:
                        running_sequence = running_sequence + 10 * 'N'
                        running_sequence = running_sequence + right_current_sequence
                        running_end = this_end

                else:
                    # we have a valid aignment so we try to find an anchor position
                    pos_a, pos_b = self.get_confident_positions(alignment)

                    if pos_a == -1 or pos_b == -1:
                        # in this case we couldn't find a place that we can use as an anchor
                        # we again compensate this Ns in the sequence.
                        sys.stderr.write(TextColor.YELLOW + "WARNING: NO OVERLAPS IN ALIGNMENT : \n" + TextColor.END)
                        sys.stderr.write(TextColor.YELLOW + "LEFT : " + str(left_running_sequence_chunk) + "\n" +
                                         TextColor.END)
                        sys.stderr.write(TextColor.YELLOW + "RIGHT: " + str(right_current_sequence) + "\n" +
                                         TextColor.END)
                        sys.stderr.write(TextColor.YELLOW + "CIGAR: " + str(alignment.cigar_string) + "\n" +
                                         TextColor.END)
                        if len(this_sequence) > 10:
                            left_sequence = running_sequence[:-overlap_bases]
                            overlap_sequence = left_running_sequence_chunk
                            running_sequence = left_sequence + overlap_sequence + 10 * 'N' + this_sequence
                            running_end = this_end
                    else:
                        # this is a perfect match so we can simply stitch them
                        # take all of the sequence from the left
                        left_sequence = running_sequence[:-overlap_bases]
                        # get the bases that overlapped
                        overlap_sequence = left_running_sequence_chunk[:pos_a]
                        # get sequences from current sequence
                        right_sequence = this_sequence[pos_b:]

                        # now append all three parts and we have a contiguous sequence
                        running_sequence = left_sequence + overlap_sequence + right_sequence
                        running_end = this_end
            else:
                # in this case we encountered a region where there's no high level overlap.
                # In this case we again compensate with Ns.
                sys.stderr.write(TextColor.YELLOW + "WARNING: NO OVERLAP IN CHUNKS: " + " " + str(contig)
                                 + " " + str(this_start) + " " + str(running_end) + "\n" + TextColor.END)

                # if the sequence is worth adding, then we add
                if len(this_sequence) > 10:
                    running_sequence = running_sequence + 10 * 'N' + this_sequence
                    running_end = this_end

        return contig, running_start, running_end, running_sequence

    def small_chunk_stitch(self, contig, small_chunk_keys):
        """
        This process stitches regional image chunk predictions. Among these image chunks, the positions are
        always consistent, we don't need an alignment to stitch these chunks.
        :param contig: Contig name
        :param small_chunk_keys: Chunk keys in list as (contig_name, start_position, end_position)
        :return:
        """
        # for chunk_key in small_chunk_keys:
        name_sequence_tuples = list()

        # go through all the chunk keys
        for contig_name, file_name, chunk_name, contig_start, contig_end in small_chunk_keys:
            smaller_chunks = []
            with h5py.File(file_name, 'r') as hdf5_file:
                if 'predictions' in hdf5_file:
                    smaller_chunks = set(hdf5_file['predictions'][contig][chunk_name].keys()) - \
                                     {'contig_start', 'contig_end'}

            smaller_chunks = sorted(smaller_chunks)
            all_positions = set()
            # now create two dictionaries where we will save all the predictions
            base_prediction_dict = defaultdict()
            rle_prediction_dict = defaultdict()
            for chunk in smaller_chunks:
                # grab the predictions and the positions
                with h5py.File(file_name, 'r') as hdf5_file:
                    bases = hdf5_file['predictions'][contig][chunk_name][chunk]['bases'][()]
                    rles = hdf5_file['predictions'][contig][chunk_name][chunk]['rles'][()]
                    positions = hdf5_file['predictions'][contig][chunk_name][chunk]['position'][()]

                positions = np.array(positions, dtype=np.int64)
                base_predictions = np.array(bases, dtype=np.int)
                rle_predictions = np.array(rles, dtype=np.int)

                # now iterate over each position and add the predictions to the dictionary
                for position, base_pred, rle_pred in zip(positions, base_predictions, rle_predictions):
                    indx = position[1]
                    pos = position[0]
                    split_indx = position[2]
                    if indx < 0 or pos < 0:
                        continue
                    if (pos, indx, split_indx) not in base_prediction_dict:
                        base_prediction_dict[(pos, indx, split_indx)] = base_pred
                        rle_prediction_dict[(pos, indx, split_indx)] = rle_pred
                        all_positions.add((pos, indx, split_indx))

            # now simply create a position list and query the  dictionary to generate the predicted sequence
            pos_list = sorted(list(all_positions), key=lambda element: (element[0], element[1], element[2]))
            dict_fetch = operator.itemgetter(*pos_list)
            predicted_base_labels = list(dict_fetch(base_prediction_dict))
            predicted_rle_labels = list(dict_fetch(rle_prediction_dict))
            sequence = ''.join([StitchOptions.label_decoder[base] * int(rle)
                                for base, rle in zip(predicted_base_labels, predicted_rle_labels)])
            # now add the generated sequence for further stitching
            name_sequence_tuples.append((contig, contig_start, contig_end, sequence))

        # now we have all the regional sequences generated we can add them using ssw.
        name_sequence_tuples = sorted(name_sequence_tuples, key=lambda element: (element[1], element[2]))

        # stich using ssw and return the contiguous sequence from the regional chunks we got from here.
        contig, running_start, running_end, running_sequence = self.alignment_stitch(name_sequence_tuples)

        return contig, running_start, running_end, running_sequence

    def create_consensus_sequence(self, contig, sequence_chunk_keys, threads):
        """
        This is the consensus sequence create method that creates a sequence for a given contig.
        :param contig: Contig name
        :param sequence_chunk_keys: All the chunk keys in the contig
        :param threads: Number of available threads
        :return: A consensus sequence for a contig
        """
        # first we sort the sequence chunks
        sequence_chunk_key_list = list()

        # then we split the chunks to so get contig name, start and end positions so we can sort them properly
        for hdf5_file, chunk_key, st, end in sequence_chunk_keys:
            sequence_chunk_key_list.append((contig, hdf5_file, chunk_key, int(st), int(end)))

        # we sort based on positions
        sequence_chunk_key_list = sorted(sequence_chunk_key_list, key=lambda element: (element[3], element[4]))

        sequence_chunks = list()
        # we submit the chunks in process pool
        with concurrent.futures.ProcessPoolExecutor(max_workers=threads) as executor:
            # this chunks the keys into sucessive chunks
            file_chunks = FileManager.chunks(sequence_chunk_key_list,
                                             max(StitchOptions.MIN_SEQUENCE_REQUIRED_FOR_MULTITHREADING,
                                                 int(len(sequence_chunk_key_list) / threads) + 1))

            # we do the stitching per chunk of keys
            futures = [executor.submit(self.small_chunk_stitch, contig, file_chunk)
                       for file_chunk in file_chunks]

            # as they complete we add them to a list
            for fut in concurrent.futures.as_completed(futures):
                if fut.exception() is None:
                    contig, contig_start, contig_end, sequence = fut.result()
                    sequence_chunks.append((contig, contig_start, contig_end, sequence))
                else:
                    sys.stderr.write("ERROR: " + str(fut.exception()) + "\n")
                fut._result = None  # python issue 27144

        sequence_chunks = sorted(sequence_chunks, key=lambda element: (element[1], element[2]))

        # and do a final stitching on all the sequences we generated
        contig, contig_start, contig_end, sequence = self.alignment_stitch(sequence_chunks)

        return sequence
