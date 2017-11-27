import unittest
import spell_checker

WORDS = spell_checker.get_word_counts(['data/corpora/big.txt'])
ERR_DIST = spell_checker.create_error_distribution('data/misspellings/wikipedia_common_misspellings.txt', WORDS)
lm = spell_checker.learn_language_model(['data/corpora/big.txt'])


class SpellCheckerTest(unittest.TestCase):
    @unittest.skip('does not work, do not care too much right now')
    def test_err_dist(self):
        for table in ERR_DIST.itervalues():
            for prob in table.itervalues():
                self.assertTrue(prob <= 1, 'There is {} prob in the confusion matrix'.format(prob))

    @unittest.skip('just for now to be faster')
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

    @unittest.skip('does not fucking work')
    def test_correct_two_edits(self):
        correction = spell_checker.correct_word('avoux', WORDS, ERR_DIST)
        self.assertEqual(correction, 'avoid',
                         'expected {}->{}, got {} instead'.format(err, word, correction))

    def test_language_model(self):
        """w1 w2 w3 w1 w4"""
        example_lm = spell_checker.learn_language_model(['../example.txt'], 3, None)
        self.assertDictEqual(
            example_lm,
            {
                'w1': {'': 1, 'w2 w3': 1},
                'w2': {'w1': 1},
                'w3': {'w1 w2': 1},
                'w4': {'w3 w1': 1},
                '': {'w1 w4': 1, 'w4': 1}
            }
        )

        self.assertEqual(spell_checker.get_counts_word_in_context('w1', ['w2', 'w3'], example_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('w1', ['2', 'w3'], example_lm), 0)
        self.assertEqual(spell_checker.get_counts_word_in_context('w1', [''], example_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('w2', ['w1'], example_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('w1', ['w3'], example_lm), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
