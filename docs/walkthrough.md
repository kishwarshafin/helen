
# MarginPolish-HELEN walkthough

This document provides a guideline on how to use the polishing pipeline.

## Installation
Please make sure you have installed `MarginPolish` and `HELEN` using the [installation guide](installation.md).


## Download data
Create a data directory and download the reads there.
```bash
mkdir -p walkthrough/
cd walkthrough
wget https://s3-us-west-2.amazonaws.com/lc2019/shasta/ecoli_test/r94_ec_rad2.181119.60x-10kb.fasta.gz
gunzip r94_ec_rad2.181119.60x-10kb.fasta.gz
```
## Assembly and Alignment
`MarginPolish` requires a draft genome assembly and a mapping of the reads to the draft assembly.

#### Generate draft assembly using Shasta
Although any assembler can be used to generate the initial assembly, we highly recommend using [Shasta](https://github.com/chanzuckerberg/shasta).

As our set of reads is very small. We can safely assemble the reads. First, download the Shasta binary.
```bash
wget https://github.com/chanzuckerberg/shasta/releases/download/0.1.0/shasta-Linux-0.1.0
chmod ugo+x shasta-Linux-0.1.0
./shasta-Linux-0.1.0 --help
```

Now assemble the reads using Shasta.
```bash
sudo ./shasta-Linux-0.1.0 --input r94_ec_rad2.181119.60x-10kb.fasta --output r94_ec_shasta_assembly
# this generates a file called "Assembly.fasta" which contains the assembly of the reads.
```

#### Create read to assembly mapping using MiniMap2
Install Minimap2
```bash
# clone the github repo and install minimap2
git clone https://github.com/lh3/minimap2
cd minimap2 && make
cd ..
```

Install Samtools
```bash
# install samtools
# install dependencies
sudo apt-get install zlib1g-dev libbz2-dev liblzma-dev
sudo apt-get install libncurses5-dev libncursesw5-dev

wget https://github.com/samtools/samtools/releases/download/1.9/samtools-1.9.tar.bz2
tar -xvjf samtools-1.9.tar.bz2
cd samtools-1.9
./configure
make
# you may need sudo permission for this
make install
cd ..
```

Suppose you have 32 threads. The command would be:
```bash
minimap2/minimap2 -ax map-ont -t 32 r94_ec_shasta_assembly/Assembly.fasta r94_ec_rad2.181119.60x-10kb.fasta | samtools sort -@ 32 | samtools view -hb -F 0x104 > reads_2_shasta_ec.bam
samtools index -@32 reads_2_shasta_ec.bam
```

## Run MarginPolish
```bash
sudo docker pull tpesout/margin_polish:latest
sudo docker run tpesout/margin_polish:latest --help
wget https://raw.githubusercontent.com/UCSC-nanopore-cgl/MarginPolish/master/params/allParams.np.human.guppy-ff-235.json
mkdir marginpolish_output
```

```bash
# Copy the absolute path to walkthough
pwd
# copy the output
```

```bash
sudo docker run -v <absolute/path/to/walkthrough/>:/data tpesout/margin_polish:latest \
 reads_2_shasta_ec.bam \
r94_ec_shasta_assembly/Assembly.fa \
allParams.np.ecoli.json \
-t 32 \
-o marginpolish_output/marginpolish_images \
-f
```
### Run HELEN
```bash
sudo docker pull kishwars/helen:0.0.1.cpu
sudo docker run kishwars/helen:0.0.1.cpu call_consensus.py -h
mkdir helen_output
wget https://storage.googleapis.com/kishwar-helen/helen_trained_models/v0.0.1/r941_flip235_v001.pkl
```

```bash
# Run call_consensus
sudo docker run -v <absolute/path/to/walkthrough/>:/data kishwars/helen:0.0.1.cpu call_consensus.py \
-i marginpolish_output/ \
-b 64 \
-m r941_flip235_v001.pkl \
-o helen_output \
-w 0 \
-t 1

# Run Stitch
sudo docker run -v <absolute/path/to/walkthrough/>:/data kishwars/helen:0.0.1.cpu \
stitch.py \
-i <helen_output/helen_predictions_XX.hdf> \
-t 32 \
-o helen_output/ \
-p r94_ec_shasta_mp_helen
```

### Benchmarking
After running Shasta-MarginPolish-HELEN, we get a polished assembly at `helen_output/r94_ec_shasta_mp_helen.fa`. We can now benchmark the assembly using `Pomoxis`.

##### Download truth Assembly
```bash
wget https://storage.googleapis.com/kishwar-friday-data/helen_polisher/e_coli_data/e_coli_truth.fasta
mkdir pomoxis_output
```

##### Install Pomoxis
```bash
git clone --recursive https://github.com/nanoporetech/pomoxis
cd pomoxis
make install
. ./venv/bin/activate
cd ..
```

```bash
# ASSESSMENT OF SHASTA ASSEMBLY
time assess_assembly \
-i r94_ec_shasta_assembly/Assembly.fasta  \
-r e_coli_truth.fasta \
-p pomoxis_output/shasta_ec_q \
-t 32
# expected error rate(err_ont): ~2.50%

time assess_assembly \
-i helen_output/r94_ec_shasta_mp_helen.fa \
-r e_coli_truth.fasta \
-p pomoxis_output/shasta_helen_ec_q \
-t 32
# expected error rate(err_ont): ~1.50%
```
