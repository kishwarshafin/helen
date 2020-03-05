
# MarginPolish-HELEN walkthough  (Docker)

This document provides a guideline on how to use the polishing pipeline.

## Docker walkthrough
This provides a guideline on how to use the docker images.

##### Setup input-output directory
```bash
mkdir mp_helen_walkthough
cd mp_helen_walkthough

mkdir mp_helen_models
mkdir mp_output
mkdir helen_output

export INPUT_DIR="$PWD"
export MODEL_OUTPUT_DIR="mp_helen_models"
export MP_OUTPUT_DIR="mp_output"
export HELEN_OUTPUT_DIR="helen_output"
```

##### Download data
The total download size is `~1.6GB`.
```bash
wget https://storage.googleapis.com/kishwar-helen/bacterial_data/guppy_305/validation_data/Reads_to_assembly_StaphAur.bam
wget https://storage.googleapis.com/kishwar-helen/bacterial_data/guppy_305/validation_data/Reads_to_assembly_StaphAur.bam.bai
wget https://storage.googleapis.com/kishwar-helen/bacterial_data/guppy_305/validation_data/Draft_assembly_StaphAur.fasta
```


##### STEP 1: Download models
First download and save all the available models.
```bash
docker run \
-v "$INPUT_DIR":"/data/" \
--rm -it --ipc=host --user="$(id -u):$(id -g)" \
kishwars/helen:latest \
helen download_models \
--output_dir "$MODEL_OUTPUT_DIR"
```

##### STEP 2: Run MarginPolish
Next, run `MarginPolish` to generate images for `HELEN`.
```bash
docker run \
-v "$INPUT_DIR":"/data/" \
--rm -it --ipc=host --user="$(id -u):$(id -g)" \
kishwars/helen:latest \
marginpolish \
"Reads_to_assembly_StaphAur.bam" \
"Draft_assembly_StaphAur.fasta" \
"$MODEL_OUTPUT_DIR/MP_r941_guppy344_microbial.json" \
-t 38 \
-o "$MP_OUTPUT_DIR/mp_images" \
-f
```

##### STEP 3: Run HELEN
Finally, run `HELEN` to get the polished sequence:
```bash
docker run \
-v "$INPUT_DIR":"/data/" \
--rm -it --ipc=host --user="$(id -u):$(id -g)" \
kishwars/helen:latest \
helen polish \
--image_dir "$MP_OUTPUT_DIR" \
--model_path "$MODEL_OUTPUT_DIR/HELEN_r941_guppy344_microbial.pkl" \
--threads 38 \
--output_dir "$HELEN_OUTPUT_DIR/" \
--output_prefix Staph_Aur_draft_helen
```

## Assessment of the assembly (Optional)
You can assess the assembly using `Pomoxis`. Make sure you exit the `venv` of `HELEN`.

### Install Pomoxis
```bash
sudo apt-get install cmake wget bzip2 zlib1g-dev libncurses5-dev \
python3-all-dev libhdf5-dev libatlas-base-dev libopenblas-base \
libopenblas-dev libbz2-dev liblzma-dev libffi-dev make python-virtualenv
```
```bash
git clone --recursive https://github.com/nanoporetech/pomoxis
cd pomoxis
make install
. ./venv/bin/activate
```

### Download reference quality assembly
```bash
wget https://storage.googleapis.com/kishwar-helen/bacterial_data/guppy_305/validation_data/truth_assembly_staph_aur.fasta
```

### Run assess_assembly
```bash
mkdir pomoxis_assessment
cd pomoxis_assessment

assess_assembly \
-i ../Draft_assembly_StaphAur.fasta \
-r ../truth_assembly_staph_aur.fasta \
-p draft_assembly_quality \
-l 50 \
-t 32 \
-e \
-T
```

Expected Output:
```bash
#  Q Scores
  name     mean      q10      q50      q90
 err_ont  24.16      inf    24.59    22.74
 err_bal  24.15      inf    24.58    22.71
    iden  31.61      inf    32.66    30.85
     del  32.10      inf    34.61    31.46
     ins  25.95      inf    26.38    23.00
```
Now assess the polished assembly:
```bash
assess_assembly \
-i ../helen_output/Staph_Aur_draft_helen.fa \
-r ../truth_assembly_staph_aur.fasta \
-p polished_assembly_quality \
-l 50 \
-t 32 \
-e \
-T
```

Expected output:
```bash
#  Q Scores
  name     mean      q10      q50      q90
 err_ont  33.90    35.09    33.98    32.84
 err_bal  33.90    35.09    33.98    32.84
    iden  38.88    40.97    39.19    37.21
     del  39.19    42.44    39.21    37.59
     ins  38.03    40.09    37.45    36.93
```
