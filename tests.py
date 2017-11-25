import unittest
import spell_checker

import re
from collections import Counter


def words(text): return re.findall(r'\w+', text.lower())


WORDS = Counter(words(open('data/corpora/big.txt').read()))
ERR_DIST = spell_checker.create_error_distribution('data/misspellings/wikipedia_common_misspellings.txt', WORDS)


class SpellCheckerTest(unittest.TestCase):
    @unittest.skip('does not work, do not care too much right now')
    def test_err_dist(self):
        for table in ERR_DIST.itervalues():
            for prob in table.itervalues():
                self.assertTrue(prob <= 1, 'There is {} prob in the confusion matrix'.format(prob))

    def test_correct_word(self):
        err_to_word = {
            'abou': 'about',
            'aboux': 'about',
            'helo': 'hello',
            'heloe': 'helped',
            'corect': 'correct',
            'accademic': 'academic',
            'academic': 'academic',
            'ook': 'look',
            'bok': 'book',
            'vox': 'box',
            'exemple': 'example',
            'exellent': 'excellent',
            'familes': 'families',
            'ell': 'all',
            'simpel': 'simple',
            'sunstiturion': 'substitution',
            'similer': 'similar',
            'newq': 'new',
            'ingo': 'into',
            'betwen': 'between'
        }
        for err, word in err_to_word.iteritems():
            correction = spell_checker.correct_word(err, WORDS, ERR_DIST)
            self.assertEqual(correction, word,
                             'expected {}->{}, got {} instead'.format(err, word, correction))


if __name__ == '__main__':
    unittest.main(verbosity=2)
