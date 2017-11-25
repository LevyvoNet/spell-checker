import unittest
import spell_checker

import re
from collections import Counter


def words(text): return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open('data/corpora/big.txt').read()))
ERR_DIST = spell_checker.create_error_distribution('data/misspellings/wikipedia_common_misspellings.txt', WORDS)


class SpellCheckerTest(unittest.TestCase):
    def test_err_dist(self):
        for table in ERR_DIST.itervalues():
            for prob in table.itervalues():
                self.assertTrue(prob <= 1, 'There is {} prob in the confusion matrix'.format(prob))

    def test_correct_word(self):
        correct_abou = spell_checker.correct_word('abou', WORDS, ERR_DIST)
        self.assertEqual(correct_abou, 'about',
                         'expected abou->about, got {} instead'.format(correct_abou))


if __name__ == '__main__':
    unittest.main(verbosity=2)
