"""
Picks a random MNIST test set digit and runs encryption and Quadratic
evaluation on it.
"""
import setup

from core import (
    discretelogarithm,
    make_keys,
    models,
    scheme
)
import numpy as np
import os
import random
from sklearn.datasets import fetch_mldata

inst = 'objects/instantiations/MNT159.inst'
vector_length = 784
k = 250
classes = 10
model = 'objects/ml_models/final.mlm'
mnist = fetch_mldata('MNIST original')
X, y = mnist["data"].astype('float'), mnist["target"].astype('float')
X_test, y_test = X[60000:], y[60000:]

index = random.randint(0, 9999)
X, y = X_test[index], y_test[index]

print('Will test on MNIST test instance #{}, which is a {}.'.format(
    index,
    int(y)
))
print('Importing model.')
ml = models.MLModel(source=model)
biased = np.ones(785)
biased[1:] = X
print('Done!\n')

results = ml.evaluate(biased)
print('Expected output:\n{}\n'.format(results))

print('Importing scheme.')

scheme = scheme.ML_DGP(inst)
print('Done!\n')

print('Loading discrete logarithm solver.')

dlog = discretelogarithm.PreCompBabyStepGiantStep(
    scheme.group,
    scheme.gt,
    minimum=-1.7e+11,
    maximum=2.7e+11,
    step=1 << 13,
)

scheme.set_dlog(
    dlog
)
print('Done!\n')

print('Importing keys...')
pk = models.PublicKey(
    source='objects/pk/common_{}.pk'.format(vector_length)
)
msk = models.MasterKey(
    source='objects/msk/common_{}.msk'.format(vector_length)
)
print('Done!\n')

print('Encrypting...')
v = models.Vector(array=X)
c = scheme.encrypt(pk, v)
print('Done!\n')

print('Generating functional decryption key...')
dk = scheme.keygen(msk, ml)
print('Done!\n')

print('Decrypting...')
dec = scheme.decrypt(pk, dk, c)
print('Done!\n')

print('Decryption result:\n{}'.format(dec))
print('Image is believed to be a {}.'.format(np.argmax(dec)))

assert (dec == results.astype(int)).all(), 'Error, decryption failed'
