# Install MarginPolish and HELEN
`MarginPolish` and `HELEN` can be used in Linux-based system. Users can install `MarginPolish` and `HELEN` on <b>`Ubuntu 18.04`</b> by following this document.

<center>
<h2> Install MarginPolish </h2>
</center>

To install MarginPolish in a Ubuntu/Linux bases system, follow these instructions:

##### Install Dependencies
```bash
sudo apt-get -y install cmake make gcc g++ autoconf bzip2 lzma-dev zlib1g-dev \
libcurl4-openssl-dev libpthread-stubs0-dev libbz2-dev liblzma-dev libhdf5-dev \
python3-pip python3-virtualenv virtualenv
```

```bash
git clone https://github.com/kishwarshafin/helen.git
cd helen
make install
. ./venv/bin/activate

marginPolish --version
helen --version
```

#### Usage
If you have installed sucessfully then please follow the [Local Install Usage Guide](docs/usage_local_install.md)
