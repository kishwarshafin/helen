from build import HELEN
from helen.modules.python.Options import StitchOptions
from collections import defaultdict


class PairWiseAlignment:

    def perform_pairwise_alignment(self, ref, query):

        aligner = HELEN.Aligner(StitchOptions.MATCH_PENALTY, StitchOptions.MISMATCH_PENALTY,
                                StitchOptions.GAP_PENALTY, StitchOptions.GAP_EXTEND_PENALTY)

        # first create a k-mer index for the ref, like minimap
        ref_kmers = defaultdict(list)
        for i in range(len(ref)):
            kmer = ref[i:i+StitchOptions.KMER_SIZE]

            if len(kmer) < StitchOptions.KMER_SIZE:
                break

            ref_kmers[kmer].append(i)

        previous_match_positions = []
        for query_index in range(0, len(query), 5):
            query_kmer = query[query_index:query_index+StitchOptions.KMER_SIZE]

            if len(query_kmer) < StitchOptions.KMER_SIZE:
                break
            ref_index_matches = ref_kmers[query_kmer]

            for ref_index in ref_index_matches:
                if ref_index - 5 not in previous_match_positions:
                    print("UNIQUE", query_index, ref_index)
                    sub_query_seq = query[query_index:]
                    sub_ref_seq = ref[ref_index:]

                    alignment = HELEN.Alignment()
                    aligner.SetReferenceSequence(sub_ref_seq, len(sub_ref_seq))
                    filter_profile = HELEN.Filter()
                    aligner.Align_cpp(sub_query_seq, filter_profile, alignment, 0)

                    print(sub_ref_seq)
                    print(sub_query_seq)
                    print(alignment.best_score)
                    print(alignment.cigar_string)
                    print("-----------------")

            previous_match_positions = ref_index_matches

        exit()




        # initialize an alignment object

        # align current sequence to the previous sequence
        filter_profile = HELEN.Filter()
        aligner.Align_cpp(query, filter_profile, alignment, 0)

        print(alignment.cigar_string)
        print(alignment.best_score)
        print(alignment.reference_begin)
        print(alignment.query_begin)


