import numpy as np
from tensorflow.python.platform import test

try:
    from tensorflow_onmttok.python.ops.onmttok_ops import detokenize, tokenize
except ImportError:
    from onmttok_ops import detokenize, tokenize


class DetokenizeTest(test.TestCase):
    def testModes(self):
        with self.session():
            self.assertAllEqual(detokenize([b"Mary-Ann", b"is", b"here", b"."], mode='conservative'),
                                np.array([b"Mary-Ann is here ."]))

            self.assertAllEqual(detokenize([b"Mary", b"-", b"Ann", b"is", b"here", b"."], mode='aggressive'),
                                np.array([b"Mary - Ann is here ."]))

            self.assertAllEqual(detokenize([b"T", b"o", b"m"], mode='char'),
                                np.array([b"T o m"]))

            self.assertAllEqual(detokenize([b"Mary-Ann", b"is", b"here."], mode='space'),
                                np.array([b"Mary-Ann is here."]))


class TokenizeTest(test.TestCase):
    def testModes(self):
        with self.session():
            self.assertAllEqual(tokenize([b"Mary-Ann is here."], mode='conservative'),
                                np.array([b"Mary-Ann", b"is", b"here", b"."]))

            self.assertAllEqual(tokenize([b"Mary-Ann is here."], mode='aggressive'),
                                np.array([b"Mary", b"-", b"Ann", b"is", b"here", b"."]))

            self.assertAllEqual(tokenize([b"Tom"], mode='char'),
                                np.array([b"T", b"o", b"m"]))

            self.assertAllEqual(tokenize([b"Mary-Ann is here."], mode='space'),
                                np.array([b"Mary-Ann", b"is", b"here."]))


if __name__ == '__main__':
    test.main()
