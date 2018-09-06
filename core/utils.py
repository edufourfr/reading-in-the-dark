"""
Contains various basic utilities used throughout the project.
"""
from base64 import b64encode, b64decode
from charm.core.engine.util import (
    serializeObject,
    deserializeObject,
    to_json,
    from_json
)
from charm.toolbox.pairinggroup import PairingGroup
import collections
import json
import numpy as np
import os
import zlib


# Base class handles some of the trouble of saving objects to a file.
class Serializable:
    ext_ = 'ser'

    def serialize(self):
        d = self.__dict__.copy()
        if 'group' in d:
            del d['group']
            return {'param': self.group.param,
                    'rest': serializeObject(d, self.group)}
        else:
            return d

    def deserialize(self, o):
        assert type(o) == dict, 'Wrong file format.'
        if 'param' in o:
            self.group = PairingGroup(o['param'])
            assert 'rest' in o, 'Wrong file format.'
            d = deserializeObject(o['rest'], self.group)
        else:
            d = o
        for key, value in d.items():
            setattr(self, key, value)
        return self

    def toFile(self, path, title):
        serialized = self.serialize()
        js = json.dumps(serialized, default=to_json)
        by = bytes(js, 'utf-8')
        compressed = zlib.compress(by)
        encoded = b64encode(compressed)
        final = encoded.decode('utf-8')
        with open(
            os.path.join(
                path,
                '{}.{}'.format(title, self.ext_),
            ),
            'w',
        ) as f:
            f.write(final)

    def fromFile(self, source):
        assert isinstance(source, str)
        assert os.path.isfile(source)
        with open(source, 'r') as f:
            final = f.read()
            encoded = bytes(final, 'utf-8')
            compressed = b64decode(encoded)
            by = zlib.decompress(compressed)
            js = by.decode('utf-8')
            serialized = json.loads(js, object_hook=from_json)
            return self.deserialize(serialized)


# Checking functions
def is_array(a):
    return isinstance(a, (collections.Sequence, np.ndarray))


def is_scalar(s):
    return (isinstance(s, int) or
            isinstance(s, np.int64) or
            (isinstance(s, float) and s.is_integer()))


# Redefine exponentiation to avoid "pairing.Error: undefined exponentiation
# operation." on non unit negative integers
def sign(x):
    return (1, -1)[x < 0]


def exp(g, x):
    return (g ** abs(x)) ** sign(x)


# Batch exponentiation functions

# Precompute every exponentiation between xmin and xmax once and for all,
# then use the results to compute the exponentiations in X without introducing
# obvious timing issues
def fast_exp_const_time(g, X, xmin=0, xmax=255):
    curr = exp(g, xmin)
    precomp = [curr]
    for i in range(xmax-xmin):
        curr *= g
        precomp.append(curr)
    return [precomp[x-xmin] for x in X]


# Compute the g^X where X are ints of at most nbits bits.
# NOT CONSTANT TIME
def batch_exp(g, X, nbits):
    one = g ** 0
    powers = [g]
    curr = g
    for b in range(nbits-1):
        curr = curr * curr
        powers.append(curr)
    out = []
    for x in X:
        bit = 1
        curr = one
        val = abs(x)
        for b in range(nbits):
            if val & bit:
                curr *= powers[b]
            bit *= 2
        if x < 0:
            curr = 1 / curr
        out.append(curr)
    return out
