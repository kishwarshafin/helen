
# MarginPolish-HELEN

### Setup
```
sudo apt-get update

sudo apt install docker.io
sudo systemctl start docker
sudo systemctl enable docker

sudo groupadd docker
sudo gpasswd -a $USER docker
# now log out and log in once or run:
newgrp docker

```

### Download data
```bash
wget -N https://s3-us-west-2.amazonaws.com/lc2019/shasta/ecoli_test/r94_ec_rad2.181119.60x-10kb.fasta.gz
gunzip r94_ec_rad2.181119.60x-10kb.fasta.gz
```

### Shasta
```bash
docker run -it --rm --user=`id -u`:`id -g` --cpus="32" -v `pwd`:/data tpesout/shasta@sha256:048f180184cfce647a491f26822f633be5de4d033f894ce7bc01e8225e846236 --input r94_ec_rad2.181119.60x-10kb.fasta
mv ShastaRun/Assembly.fasta shasta.fasta
```

### minimap2
```bash
docker run -it --rm --user=`id -u`:`id -g` --cpus="32" -v `pwd`:/data tpesout/minimap2@sha256:5df3218ae2afebfc06189daf7433f1ade15d7cf77d23e7351f210a098eb57858 -ax map-ont -t 32 shasta.fasta r94_ec_rad2.181119.60x-10kb.fasta
```

### samtools
```bash
docker run -it --rm --user=`id -u`:`id -g` --cpus="32" -v `pwd`:/data tpesout/samtools_sort:latest /data/minimap2.sam -@ 32
docker run -it --rm --user=`id -u`:`id -g` --cpus="32" -v `pwd`:/data tpesout/samtools_view@sha256:11faa9b074b3ec96f50f62133bd19f819bd5bf6ad879d913ac45955f95dd91fb -hb -F 0x104 /data/samtools_sort.bam
docker run -it --rm --user=`id -u`:`id -g` --cpus="32" -v `pwd`:/data quay.io/ucsc_cgl/samtools:1.8--cba1ddbca3e1ab94813b58e40e56ab87a59f1997 index -@ 32 /data/samtools_sort.bam
```

### marginpolish
```
mkdir -p marginPolish
docker run -it --rm --user=`id -u`:`id -g` --cpus="32" -v `pwd`:/data tpesout/margin_polish@sha256:de10c726bcc6af2f58cbb35af32ed0f0d95a3dc5f64f66dcc4eecbeb36f98b65 /data/samtools_sort.bam /data/shasta.fasta /opt/MarginPolish/params/allParams.np.human.guppy-ff-235.json -t 32 -o /data/marginPolish/output -f
```

### helen
```bash
wget -N https://storage.googleapis.com/kishwar-helen/helen_trained_models/v0.0.1/r941_flip235_v001.pkl
docker run -it --rm --user=`id -u`:`id -g` --cpus="32" -v `pwd`:/data kishwars/helen:0.0.1.cpu call_consensus.py -i /data/marginPolish/ -m r941_flip235_v001.pkl -o helen_hdf5/ -p prediction -w 0 -t 28
docker run -it --rm --user=`id -u`:`id -g` --cpus="32" -v `pwd`:/data kishwars/helen:0.0.1.cpu stitch.py -i helen_hdf5/prediction.hdf -o /data/ -p shasta_mp_helen_assembly -t 32
```
