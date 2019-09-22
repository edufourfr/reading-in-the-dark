# Reading In The Dark: Classifying Encrypted Digits with Functional Encryption

This code implements the cryptographic part of
[**Partially Encrypted Machine Learning using Functional Encryption**](https://arxiv.org/abs/1905.10214) (to appear at
NIPS 2019). The code for the Collateral Learning component can be found
[**here**](https://github.com/LaRiffle/collateral-learning).

## Requirements

### Install PBC and Charm

Follow this [**tutorial**](https://jhuisi.github.io/charm/install_source.html).

### Set up a database

Follow the beginning of this [**tutorial**](https://www.fullstackpython.com/blog/postgresql-python-3-psycopg2-ubuntu-1604.html) to create a database and an account and update the core/discretelogarithm file.
Default is user **dbuser** with password **password** and database **discretelogarithm**.

### Hashing

Discrete log precomputation assumes a specific environment:
~~~~
export PYTHONHASHSEED=0
~~~~

### Tinker

If you want to be able to generate the confusion matrix, you will need tinker. On Ubuntu:

~~~~
sudo apt-get install python3-tk
~~~~

### Python requirements

Installing the required python packages should be as easy as:
~~~~
pip3 install -r requirements.txt
~~~~

### Generate your keys and fill you database

~~~~
cd mnist
python3 initialization.py
~~~~

This might take a while. Note that the progress bar that will appear often does not progress linearly: the first few percentage points will be filled faster than the last few.

## Running the provided code

### Testing

Start by running a simple end-to-end test.
~~~~
cd mnist
python3 ml_test.py
~~~~
This will pick a random digit from the MNIST test set, encrypt it, and evaluate the provided model on it using functional decryption.

### Generating the confusion matrix

~~~~
cd mnist
python3 confusion_matrix.py
~~~~
This operates on cleartext data.

### Benchmarking the scheme on your hardware

~~~~
cd mnist
python3 -O benchmark.py
~~~~

## Project structure

- **core**: The code for the Quadratic FE scheme, optimized for 1 hidden layer Quadratic Networks.
  * **discretelogarithm**: A discrete logarithm solver that uses the Baby Step Giant Step method and interacts with a database to store and retrieve procomputations.
  * **make_keys**: A simple key generation function.
  * **models**: Classes and methods for vectors, cryptographic keys and machine learning models.
  * **scheme**: The implementation of the Quadratic Functional Encryption scheme.
  * **utils**: Defines some utilities, e.g. for serializing objects and batch exponentiations.
- **mnist**: Code testing and benchmarking the implementation on the MNIST dataset.
  * **objects**: A directory of directories where objects are stored durably. Should initially contain an instantiation (a choice of Pairing Group with base elements) and a pretrained MNIST model.
  * **benchmark**: Runs encryption and decryption on the whole MNIST test set and reports the timings. This takes a while.
  * **confusion_matrix**: Generates the confusion matrix found in the paper.
  * **initialization**: A run once script to run before running anything else. Generates keys and fills the database with discrete logarithm precomputations, which takes some time.
  * **ml_test**: Picks a MNIST digit at random, encrypts it and functionally decrypts it.
  * **setup**: Used to make importing from core easier.

## Issues

Try opening an issue on GitHub.

## Old paper

An earlier version of this work can be found on [**eprint**](https://eprint.iacr.org/2018/206). It may be of interest
to people who seek a better understanding of this code or of the cryptography involved.
