import argparse
"""
HELPER SCRIPT TO COUNT QUAST MISASSEMBLIES THAT HAPPEN OUTSIDE KNOWN SVs.
"""


def read_quast_file(file):
    """
    Extracts misassembly regions from QUAST file, usually located in /QUAST_output/contig_reports/all_alignments_$$.tsv
    :param file:
    :return:
    """
    total_count = 0
    relocation_count = 0
    translocation_count = 0
    inversion_count = 0
    misassemblies = []
    with open(file) as f:
        prev_line = ''
        for line in f:
            line = line.rstrip().replace(',', '')
            splits = line.split('\t')
            if splits[0].split(' ')[0] == 'relocation':
                s_ref, e_ref, s_con, e_con, ref, con, idn, ambi, bg = prev_line.split('\t')
                # print(ref, s_ref, e_ref, splits[0])
                misassemblies.append([ref, s_ref, e_ref, splits[0].split(' ')[0]])
                relocation_count += 1
                total_count += 1
            elif splits[0] == 'translocation':
                s_ref, e_ref, s_con, e_con, ref, con, idn, ambi, bg = prev_line.split('\t')
                # print(ref, s_ref, e_ref, splits[0])
                misassemblies.append([ref, s_ref, e_ref, splits[0]])
                translocation_count += 1
                total_count += 1
            elif splits[0] == 'inversion':
                s_ref, e_ref, s_con, e_con, ref, con, idn, ambi, bg = prev_line.split('\t')
                # print(ref, s_ref, e_ref, splits[0])
                misassemblies.append([ref, s_ref, e_ref, splits[0]])
                inversion_count += 1
                total_count += 1
            prev_line = line

    print("#####--TOTAL MISSASSEMBLIES: REPORTED BY QUAST--#####")
    print("Total Misassemblies:\t", total_count)
    print("Total relocations:\t", relocation_count)
    print("Total translocations:\t", translocation_count)
    print("Total inversions:\t", inversion_count)
    print("#####################################################\n")
    # returns a list of misassemblies as a list of list where each element is [chr_name, st, end, type_of_missassmbly]
    return misassemblies


def read_bed_file(file):
    known_svs = []
    with open(file) as f:
        for line in f:
            line = line.rstrip()
            known_svs.append(line.split('\t')[0:3])
            # print(line.split('\t')[0:3])
    # returns a list of known svs as a list where each element is [chr_name, st, end]
    return known_svs


def count_miassemblies_in_autosomes(misassemblies):
    autosomes = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']
    total_count = 0
    relocation_count = 0
    translocation_count = 0
    inversion_count = 0

    for chr, st, end, ms_type in misassemblies:
        if chr in autosomes:
            if ms_type == 'relocation':
                relocation_count += 1
                total_count += 1
            elif ms_type == 'translocation':
                translocation_count += 1
                total_count += 1
            elif ms_type == 'inversion':
                inversion_count += 1
                total_count += 1
        else:
            print(chr, st, end, ms_type)

    print("#####--TOTAL MISSASSEMBLIES: IN AUTOSOMES--#####")
    print("Total Misassemblies:\t", total_count)
    print("Total relocations:\t", relocation_count)
    print("Total translocations:\t", translocation_count)
    print("Total inversions:\t", inversion_count)
    print("################################################\n")


def count_misassemblies_not_overlapping_with_svs(known_svs, misassemblies):
    total_count = 0
    total_base_count = 0
    relocation_count = 0
    translocation_count = 0
    inversion_count = 0
    autosomes = ['chr' + str(x) for x in range(1, 23)] + ['chrX', 'chrY']

    for chr_ms, st_ms, end_ms, ms_type in misassemblies:
        if chr_ms not in autosomes:
            continue

        overlaps = False
        for chr_sv, st_sv, end_sv in known_svs:
            # x1 <= y2 && y1 <= x2
            if chr_ms == chr_sv and int(st_ms) <= int(end_sv) and int(st_sv) <= int(end_ms):
                # print(chr_ms, st_ms, end_ms, ms_type, 'overlaps with sv', chr_sv, st_sv, end_sv)
                overlaps = True
                break
            if overlaps:
                break

        if overlaps is False:
            total_count += 1
            total_base_count += (int(end_ms) - int(st_ms) + 1)
            if ms_type == 'relocation':
                relocation_count += 1
            elif ms_type == 'translocation':
                translocation_count += 1
            elif ms_type == 'inversion':
                inversion_count += 1

    print("#####--MISSASSEMBLIES OUTSIDE SVs--#####")
    print("Total Misassemblies:\t", total_count)
    print("Total relocations:\t", relocation_count)
    print("Total translocations:\t", translocation_count)
    print("Total inversions:\t", inversion_count)
    print("Total Bases:\t", total_base_count/ 1000000000)
    print("########################################")


if __name__ == '__main__':
    '''
    Processes arguments and performs tasks.
    '''
    parser = argparse.ArgumentParser(description="Reports QUAST misassemblies that happen outside the known SVs.")
    parser.add_argument(
        "-q",
        "--quast_file",
        type=str,
        required=True,
        help="Path to the QUAST's TSV input file located in /QUAST_output/contig_reports/all_alignments_$$.tsv"
    )
    parser.add_argument(
        "-s",
        "--sv_file",
        type=str,
        required=True,
        help="Path to a bed file containing all known SVs for the sample."
    )
    parser.add_argument(
        "-c",
        "--cen_file",
        type=str,
        required=True,
        help="Path to a bed file containing all known centromeric region."
    )
    parser.add_argument(
        "-d",
        "--segdup_file",
        type=str,
        required=True,
        help="Path to a bed file containing all known segdup for the sample."
    )
    FLAGS, unparsed = parser.parse_known_args()
    # get regions from files
    svs = read_bed_file(FLAGS.sv_file)
    centromeres = read_bed_file(FLAGS.cen_file)
    segdups = read_bed_file(FLAGS.segdup_file)

    centromeres_only = centromeres
    centromeres_and_segdups = centromeres + segdups
    m_assemblies = read_quast_file(FLAGS.quast_file)

    # count stuff
    # count_miassemblies_in_autosomes(m_assemblies)
    print("OUTSIDE CENTROMERES")
    count_misassemblies_not_overlapping_with_svs(centromeres_only, m_assemblies)
    print("\nOUTSIDE CENTROMERES AND SEG DUPS")
    count_misassemblies_not_overlapping_with_svs(centromeres_and_segdups, m_assemblies)