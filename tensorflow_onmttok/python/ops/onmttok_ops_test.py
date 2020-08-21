import numpy as np
from tensorflow.python.platform import test

try:
    from tensorflow_onmttok.python.ops.onmttok_ops import detokenize, tokenize
except ImportError:
    from onmttok_ops import detokenize, tokenize


class DetokenizeTest(test.TestCase):
    def testModes(self):
        with self.session():
            self.assertAllEqual(
                detokenize([b"Mary-Ann", b"is", b"here", b"."], mode="conservative"),
                np.array([b"Mary-Ann is here ."])
            )

            self.assertAllEqual(
                detokenize([b"Mary", b"-", b"Ann", b"is", b"here", b"."], mode="aggressive"),
                np.array([b"Mary - Ann is here ."])
            )

            self.assertAllEqual(
                detokenize([b"T", b"o", b"m"], mode="char"),
                np.array([b"T o m"])
            )

            self.assertAllEqual(
                detokenize([b"Mary-Ann", b"is", b"here."], mode="space"),
                np.array([b"Mary-Ann is here."])
            )


class TokenizeTest(test.TestCase):
    def testModes(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Mary-Ann is here."], mode="conservative"),
                np.array([b"Mary-Ann", b"is", b"here", b"."])
            )

            self.assertAllEqual(
                tokenize([b"Mary-Ann is here."], mode="aggressive"),
                np.array([b"Mary", b"-", b"Ann", b"is", b"here", b"."])
            )

            self.assertAllEqual(
                tokenize([b"Tom"], mode="char"),
                np.array([b"T", b"o", b"m"])
            )

            self.assertAllEqual(
                tokenize([b"Mary-Ann is here."], mode="space"),
                np.array([b"Mary-Ann", b"is", b"here."])
            )

    def testCaseFeature(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Tom"],
                         mode="conservative",
                         case_feature=True),
                np.array([b"tom"])
            )

    def testCaseMarkup(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Tom"],
                         mode="conservative",
                         case_markup=True),
                np.array([b"\xef\xbd\x9fmrk_case_modifier_C\xef\xbd\xa0", b"tom"])
            )

    def testSoftCaseRegions(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"U.N"],
                         mode="conservative",
                         case_markup=True,
                         soft_case_regions=True),
                np.array([b"\xef\xbd\x9fmrk_begin_case_region_U\xef\xbd\xa0",
                          b"u.",
                          b"n",
                          b"\xef\xbd\x9fmrk_end_case_region_U\xef\xbd\xa0"])
            )

    def testJoinerAnnotate(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Hello!"],
                         mode="conservative",
                         joiner_annotate=True),
                np.array([b"Hello", b"\xc3\xaf\xc2\xbf\xc2\xad!"])
            )

    def testJoinerCustom(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Hello!"],
                         mode="conservative",
                         joiner_annotate=True,
                         joiner="@@"),
                np.array([b"Hello", b"@@!"])
            )

    def testJoinerNew(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Hello!"],
                         mode="conservative",
                         joiner_annotate=True,
                         joiner_new=True),
                np.array([b"Hello", b"\xc3\xaf\xc2\xbf\xc2\xad", b"!"])
            )

    def testSpacerAnnotate(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Hello world"],
                         mode="conservative",
                         spacer_annotate=True),
                np.array([b"Hello", b"\xe2\x96\x81world"])
            )

    def testSpacerNew(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Hello world"],
                         mode="conservative",
                         spacer_annotate=True,
                         spacer_new=True),
                np.array([b"Hello", b"\xe2\x96\x81", b"world"])
            )

    def testPreservePlaceholders(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"Hello \xef\xbd\x9fWorld\xef\xbd\xa0"],
                         mode="conservative",
                         joiner_annotate=True,
                         preserve_placeholders=True),
                np.array([b"Hello", b"\xef\xbd\x9fWorld\xef\xbd\xa0"])
            )

    def testPreserveSegTokens(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"\xe6\xb8\xac\xe8\xa9\xa6abc"],
                         mode="aggressive",
                         joiner_annotate=True,
                         segment_alphabet=["Han"],
                         segment_alphabet_change=True,
                         preserve_segmented_tokens=True),
                np.array([b"\xe6\xb8\xac",
                          b"\xc3\xaf\xc2\xbf\xc2\xad",
                          b"\xe8\xa9\xa6",
                          b"\xc3\xaf\xc2\xbf\xc2\xad",
                          b"abc"])
            )

    def testSupportPriorJoiners(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"pre\xc3\xaf\xc2\xbf\xc2\xad tokenization."],
                         mode="aggressive",
                         joiner_annotate=True,
                         support_prior_joiners=True),
                np.array([b"pre\xc3\xaf",
                          b"\xc3\xaf\xc2\xbf\xc2\xad\xc2\xbf",
                          b"\xc3\xaf\xc2\xbf\xc2\xad\xc2\xad",
                          b"tokenization",
                          b"\xc3\xaf\xc2\xbf\xc2\xad."])
            )

    def testSegmentCase(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"WiFi"],
                         mode="conservative",
                         segment_case=True),
                np.array([b"Wi", b"Fi"])
            )

    def testSegmentNumbers(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"1234"],
                         mode="aggressive",
                         segment_numbers=True),
                np.array([b"1", b"2", b"3", b"4"])
            )

    def testSegmentAlphabet(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"abcd"],
                         mode="conservative",
                         segment_alphabet=["Latin"]),
                np.array([b"a", b"b", b"c", b"d"])
            )

    def testSegmentAlphabetChange(self):
        with self.session():
            self.assertAllEqual(
                tokenize([b"\xe6\xb8\xac\xe8\xa9\xa6abc"],
                         mode="conservative",
                         segment_alphabet_change=True),
                np.array([b"\xe6\xb8\xac\xe8\xa9\xa6", b"abc"])
            )


if __name__ == "__main__":
    test.main()
