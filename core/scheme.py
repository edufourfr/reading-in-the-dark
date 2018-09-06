"""
Implementation of Quadratic Functional Encryption for ML models as defined in
Edouard Dufour Sans, Romain Gay, and David Pointcheval. "Reading in the Dark:
Classifying Encrypted Digits with Functional Encryption."
https://eprint.iacr.org/2018/206
"""
from charm.toolbox.pairinggroup import ZR, G1, G2, pair
from .discretelogarithm import PreCompBabyStepGiantStep
from .models import (
    DecryptionKey,
    EncryptedVector,
    MasterKey,
    MLModel,
    PublicKey
)
from .utils import exp, fast_exp_const_time, is_array, Serializable


class ML_DGP(Serializable):

    ext_ = 'inst'

    def __init__(self, source=''):
        if source:
            self.fromFile(source)
            return

    def create_(group):
        instance = ML_DGP()
        instance.group = group
        instance.g1 = instance.group.random(G1)
        instance.g2 = instance.group.random(G2)
        instance.gt = pair(instance.g1, instance.g2)
        return instance

    def get_dlog_base(self):
        return self.gt

    def set_dlog(self, dlog):
        assert dlog.base == self.gt
        self.dlog = dlog

    def setup(self, vector_length=1):
        assert vector_length > 0, 'Vector size must be positive.'
        s = [self.group.random(ZR) for i in range(vector_length+1)]
        t = [self.group.random(ZR) for i in range(vector_length+1)]
        h1 = [self.g1 ** si for si in s]
        h2 = [self.g2 ** ti for ti in t]
        pk = PublicKey(self.group, h1, h2)
        msk = MasterKey(pk=pk, s=s, t=t)
        return (pk, msk)

    def encrypt(self, pk, vector):
        assert (
            pk.n == vector.n + 1
        ), (
            'Vector has length {}, key is for length {}.'.format(
                vector.n,
                pk.n - 1
            )
        )
        gamma = self.group.random(ZR)
        a = self.group.random(ZR)
        b = self.group.random(ZR)
        c = self.group.random(ZR)
        d = self.group.random(ZR)
        det = a * d - b * c
        assert det != 0
        inv_a = d / det
        inv_b = -b / det
        inv_c = -c / det
        inv_d = a / det

        g1_inva = self.g1 ** inv_a
        g1_invb = self.g1 ** inv_b
        g2_a = self.g2 ** a
        g2_c = self.g2 ** c
        exp_g1_inva = fast_exp_const_time(g1_inva, vector.content, 0, 255)
        exp_g1_invb = fast_exp_const_time(g1_invb, vector.content, 0, 255)
        exp_g2_a = fast_exp_const_time(g2_a, vector.content, 0, 255)
        exp_g2_c = fast_exp_const_time(g2_c, vector.content, 0, 255)
        left = [  # bias term
            [
                g1_inva * (pk.h1[0] ** (gamma * inv_c)),
                g1_invb * (pk.h1[0] ** (gamma * inv_d))
            ]
        ]
        right = [  # bias term
            [
                g2_a * (pk.h2[0] ** (-b)),
                g2_c * (pk.h2[0] ** (-d))
            ]
        ]
        for i in range(1, pk.n):
            left.append(
                [
                    exp_g1_inva[i-1] * (pk.h1[i] ** (gamma * inv_c)),
                    exp_g1_invb[i-1] * (pk.h1[i] ** (gamma * inv_d))
                ]
            )
            right.append(
                [
                    exp_g2_a[i-1] * (pk.h2[i] ** (-b)),
                    exp_g2_c[i-1] * (pk.h2[i] ** (-d))
                ]
            )
        return EncryptedVector(
            group=self.group,
            simplifier=self.g1 ** gamma,
            left=left,
            right=right,
        )

    def keygen(self, msk, model):
        return DecryptionKey(
            model=model,
            skf=[self.g2 ** x for x in model.evaluate(msk)],
        )

    def decrypt(self, pk, sk_model, c):
        evaluation = sk_model.model.evaluate(c)
        decrypted = [
            evaluation[i] * pair(
                c.simplifier,
                sk_model.skf[i]
            ) for i in range(sk_model.classes)
        ]
        out = list(map(self.dlog.solve, decrypted))
        return out

    def serialize(self):
        dlog = None
        d = self.__dict__
        if 'dlog' in d:
            dlog = self.dlog
            del d['dlog']
        out = super(ML_DGP, self).serialize()
        if dlog:
            self.dlog = dlog
        return out
