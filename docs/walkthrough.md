
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

Please see the [quick start documentation](https://chanzuckerberg.github.io/shasta/QuickStart.html) to see how to use Shasta. Shasta requires memory intensive computing.
> For a human size assembly, AWS instance type x1.32xlarge is recommended. It is usually available at a cost around $4/hour on the AWS spot market and should complete the human size assembly in a few hours, at coverage around 60x.

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
# if you get dependency errors make sure you have installed dependency for MarginPolish and install make and gcc
sudo apt-get install make gcc
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
wget https://github.com/UCSC-nanopore-cgl/MarginPolish/blob/master/params/allParams.np.human.guppy-ff-235.json
```

```bash
marginPolish \
data/reads_2_assembly.bam \
data/shasta_assembly/Assembly.fasta \
../params/allParams.np.ecoli.json \
-t 32 \
-o data/marginpolish_images/marginpolish_images \
-f

sudo docker run -v "$pwd":/data tpesout/margin_polish:latest reads_2_shasta_ec.bam \
r94_ec_shasta_assembly/Assembly.fa \
allParams.np.ecoli.json \
-t 32 \
-o output/marginpolish_images \
-f
```
## Run HELEN
```bash
time python3 call_consensus.py \
--image_file data/marginpolish_images/ \
--batch_size 512 \
--model_path data/helen_models/HELEN_v0_lc_r941_flip233_hap.pkl \
--output_dir data/helen_out/consensus_sequence/ \
--num_workers 32 \
--gpu_mode 1
```

```bash
time python3 stitch.py \
--sequence_hdf data/helen_out/consensus_sequence/helen_predictions.hdf \
--output_dir data/helen_out/consensus_sequence/ \
--threads 32
```
