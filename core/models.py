"""
Defines classes that represent vectors (encrypted or not), cryptographic keys
and machine learning models as well as useful methods on them.
"""
from math import log2
import numpy as np
from .utils import batch_exp, exp, is_array, is_scalar, Serializable


# Vectors

class WrongInputError(BaseException):
    pass


class Vector(Serializable):

    ext_ = 'vec'

    def __init__(self, array=None, source=''):
        if source:
            self.fromFile(source)
            return
        if is_array(array):
            assert (
                len(array) > 0
            ), (
                'Trying to generate an image from an empty vector.'
            )
            self.n = len(array)
            self.content = []
            for s in array:
                assert is_scalar(s), "Input doesn't contain valid scalars."
                self.content.append(int(s))
        else:
            WrongInputError(array)


class EncryptedVector(Serializable):

    ext_ = 'evec'

    def __init__(
        self,
        group=None,
        simplifier=None,
        left=None,
        right=None,
        source=''
    ):
        if source:
            self.fromFile(source)
            return
        assert left
        assert right
        assert group
        assert simplifier
        assert is_array(left)
        assert is_array(right)
        assert (
            len(left) == len(right)
        ), (
            'Ciphertext was not properly generated.'
        )
        assert len(left) > 1, 'Ciphertext is empty.'
        assert (is_array(x) for x in left)
        assert (is_array(x) for x in right)
        assert (len(x) == 2 for x in left)
        assert (len(x) == 2 for x in right)
        self.n = len(left)
        self.group = group
        self.simplifier = simplifier
        self.left = left
        self.right = right

    def __pow__(self, forms):
        paired = [
            self.group.pair_prod(
                self.left[i],
                self.right[i]
            ) for i in range(self.n)
        ]
        out = [1 for cl in range(forms.classes)]
        for i in range(self.n):
            batch = batch_exp(paired[i], forms.content[i], forms.nbits[i])
            for cl in range(forms.classes):
                out[cl] *= batch[cl]
        return out

# Keys


class PublicKey(Serializable):

    ext_ = 'pk'

    def __init__(self, group=None, h1=None, h2=None, source=''):
        if source:
            self.fromFile(source)
            return
        assert group
        assert h1
        assert is_array(h1)
        assert is_array(h2)
        assert len(h1) == len(h2)
        self.group = group
        self.n = len(h1)
        self.h1 = h1
        self.h2 = h2


class MasterKey(PublicKey):

    ext_ = 'msk'

    def __init__(self, pk=None, s=None, t=None, source=''):
        if source:
            self.fromFile(source)
            return
        super(MasterKey, self).__init__(pk.group, pk.h1, pk.h2)
        assert s
        assert is_array(s)
        assert len(s) == len(pk.h1)
        assert t
        assert is_array(t)
        assert len(t) == len(pk.h1)
        self.s = s
        self.t = t


class DecryptionKey(Serializable):

    ext_ = 'dk'

    def __init__(self, model=None, skf=None, source=''):
        if source:
            self.fromFile(source)
            return
        assert model
        assert skf
        assert isinstance(model, MLModel)
        assert is_array(skf)
        classes = len(skf)
        assert classes
        self.classes = classes
        self.model = model
        self.skf = skf

    def serialize(self):
        model = self.model
        del self.model
        self.model_ = model.serialize()
        return super(DecryptionKey, self).serialize()

    def deserialize(self, o):
        super(DecryptionKey, self).deserialize(o)
        self.model = MLModel().deserialize(self.model_)
        del self.model_
        return self

# ML


class Projection(Serializable):

    ext_ = 'proj'

    def __init__(self, matrix=None, source=''):
        if source:
            self.fromFile(source)
            return
        if matrix is None:
            return
        assert is_array(matrix)
        n = len(matrix)
        assert n > 0
        assert (is_array(matrix[i]) for i in range(n))
        k = len(matrix[0])
        assert k > 0
        assert (len(matrix[i]) == k for i in range(n))
        assert (is_scalar(matrix[i][j]) for i in range(n) for j in range(k))
        self.n = n
        self.k = k
        self.columns = [
            [
                int(matrix[i][j]) for j in range(k)
            ] for i in range(n)
        ]
        self.nbits = []
        for j in range(n):
            m = 1
            for x in self.columns[j]:
                if abs(x) > m:
                    m = abs(x)
            self.nbits.append(int(log2(m)) + 1)

    def __mul__(self, X):
        if isinstance(X, EncryptedVector):
            left = [[1, 1] for i in range(self.k)]
            right = [[1, 1] for i in range(self.k)]
            for j in range(self.n):
                l1 = batch_exp(X.left[j][0], self.columns[j], self.nbits[j])
                l2 = batch_exp(X.left[j][1], self.columns[j], self.nbits[j])
                r1 = batch_exp(X.right[j][0], self.columns[j], self.nbits[j])
                r2 = batch_exp(X.right[j][1], self.columns[j], self.nbits[j])
                for i in range(self.k):
                    left[i][0] *= l1[i]
                    left[i][1] *= l2[i]
                    right[i][0] *= r1[i]
                    right[i][1] *= r2[i]
            return EncryptedVector(
                group=X.group,
                simplifier=X.simplifier,
                left=left,
                right=right,
            )
        elif is_array(X):
            import numpy as np
            assert self.n == len(X)
            return np.dot(np.transpose(self.columns), X)
        else:
            raise WrongInputError(X)


class DiagonalQuadraticForms(Serializable):

    ext_ = 'dqf'

    def __init__(self, matrix=None, source=''):
        if source:
            self.fromFile(source)
            return
        if matrix is None:
            return
        assert is_array(matrix)
        k = len(matrix)
        assert k
        assert (is_array(matrix[i]) for i in range(k))
        classes = len(matrix[0])
        assert classes
        assert (len(matrix[i]) == classes for i in range(k))
        assert (
            is_scalar(matrix[i][j]) for i in range(k) for j in range(classes)
        )
        self.classes = classes
        self.k = k
        self.content = [
            [int(matrix[i][j]) for j in range(classes)] for i in range(k)
        ]
        self.nbits = []
        for j in range(k):
            m = 1
            for x in self.content[j]:
                if abs(x) > m:
                    m = abs(x)
            self.nbits.append(int(log2(m)) + 1)


class MLModel(Serializable):

    ext_ = 'mlm'

    def __init__(self, proj=None, forms=None, source=''):
        if source:
            self.fromFile(source)
            return
        assert proj
        assert forms
        assert isinstance(proj, Projection)
        assert isinstance(forms, DiagonalQuadraticForms)
        assert forms.k == proj.k
        self.proj = proj
        self.forms = forms

    def evaluate(self, X):
        if isinstance(X, EncryptedVector):
            return (self.proj * X) ** self.forms
        elif isinstance(X, MasterKey):
            Ps = self.proj * X.s
            Pt = self.proj * X.t
            element_wise_prod = np.multiply(Ps, Pt)
            return np.dot(
                np.transpose(self.forms.content),
                element_wise_prod
            ).tolist()
        elif is_array(X):
            PXsquared = np.square(self.proj * X)
            return np.dot(np.transpose(self.forms.content), PXsquared)
        else:
            raise WrongInputError(X)

    def serialize(self):
        proj = self.proj
        forms = self.forms
        del self.proj
        del self.forms
        self.proj_ = proj.serialize()
        self.forms_ = forms.serialize()
        return super(MLModel, self).serialize()

    def deserialize(self, o):
        super(MLModel, self).deserialize(o)
        self.proj = Projection().deserialize(self.proj_)
        del self.proj_
        self.forms = DiagonalQuadraticForms().deserialize(self.forms_)
        del self.forms_
        return self

    def naive_bounds(self):
        maxes = []
        P = np.transpose(self.proj.columns)
        for j in range(self.proj.k):
            neg = sum([255 * p if p < 0 else 0 for p in P[j][1:]]) + P[j][0]
            pos = sum([255 * p if p > 0 else 0 for p in P[j][1:]]) + P[j][0]
            maxes.append(max(-neg, pos)**2)
        largest = 0
        lowest = 0
        matrix = np.transpose(self.forms.content)
        for cl in range(self.forms.classes):
            neg = sum(
                [
                    (
                        - maxes[j] * matrix[cl][j] if matrix[cl][j] < 0 else 0
                    ) for j in range(self.proj.k)
                ]
            )
            pos = sum(
                [
                    (
                        maxes[j] * matrix[cl][j] if matrix[cl][j] > 0 else 0
                    ) for j in range(self.proj.k)
                ]
            )
            if neg > lowest:
                lowest = neg
            if pos > largest:
                largest = pos
        return (-lowest, largest)

    def get_accuracy(self):
        assert self.proj.n == 785, 'get_accuracy only works for mnist'
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets("/tmp/data/")
        X_test_ = np.round(255*mnist.test.images)
        y_test = mnist.test.labels.astype("int")
        X_test = np.ones((10000, 785))
        X_test[:, 1:] = X_test_
        good = 0
        for i in range(10000):
            good += 1 if (
                np.argmax(self.evaluate(X_test[i])) == y_test[i]
            ) else 0
        print('Accuracy: {}%'.format(good/100))
