//
// Created by Kishwar Shafin on 25/2/19.
//

#ifndef HELEN_SUMMARY_GENERATOR_H
#define HELEN_SUMMARY_GENERATOR_H

#include <math.h>
#include <algorithm>
#include <iomanip>
#include <assert.h>
using namespace std;
#include "../dataio/bam_handler.h"
#include "../candidate_finding/candidate_finder.h"

class SummaryGenerator {
    long long ref_start;
    long long ref_end;
    string chromosome_name;
    string reference_sequence;

    map< pair< pair<long long, int>, int>, double> insert_summaries;
    map< pair<long long, int>, double> base_summaries;
    map<long long, long long> longest_insert_count;
    map<long long, long long> coverage;

    map< pair<long long, int>, char> insert_labels;
    map< long long, char> base_labels;
public:
    vector< vector<double> > image;
    vector<int> labels;
    vector<pair<long long, int> > genomic_pos;

    SummaryGenerator(string reference_sequence,
                   string chromosome_name,
                   long long ref_start,
                   long long ref_end);
    void generate_summary(vector<type_read> &reads,
                          long long start_pos,
                          long long end_pos);

    void generate_train_summary(vector <type_read> &reads,
                                long long start_pos,
                                long long end_pos,
                                type_read truth_read,
                                bool train_mode);

    void iterate_over_read(type_read read, long long region_start, long long region_end);
    int get_sequence_length(long long start_pos, long long end_pos);
    void generate_labels(type_read truth_reads, long long region_start, long long region_end);
    void debug_print(long long start_pos, long long end_pos);
};


#endif //HELEN_SUMMARY_GENERATOR_H