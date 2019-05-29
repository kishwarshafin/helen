import pandas as pd
from collections import defaultdict

medaka_table = pd.read_csv('/data/users/kishwar/marginpolish_output/bac_analysis/medaka_hg00733_bac_stats.txt', sep='\t')


def is_overlapping(x1,x2,y1,y2):
    return max(x1,y1) <= min(x2,y2)


alignment_dict = defaultdict()
alignment_intervals = []
count = 0
for index, row in medaka_table.iterrows():
    if float(row['coverage']) == 100.0:
        count += 1
        alignment_dict[str(row['name'])] = True
        alignment_intervals.append((str(row['name']), str(row['ref']), int(row['rstart']), int(row['rend'])))


for i in range(len(alignment_intervals)):
    for j in range(len(alignment_intervals)):
        if i == j:
            continue
        if alignment_intervals[i][1] != alignment_intervals[j][1]:
            continue
        if is_overlapping(alignment_intervals[i][2], alignment_intervals[j][2],
                          alignment_intervals[i][3], alignment_intervals[j][3]):
            alignment_dict[alignment_intervals[i][0]] = False
            alignment_dict[alignment_intervals[j][0]] = False

for i, alignment_name in enumerate(alignment_dict.keys()):
    if alignment_dict[alignment_name] is True:
        print(i, alignment_name)
print(count)
