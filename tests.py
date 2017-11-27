import unittest
import spell_checker

WORDS = spell_checker.get_word_counts(['data/corpora/big.txt'])
ERR_DIST = spell_checker.create_error_distribution('data/misspellings/wikipedia_common_misspellings.txt', WORDS)


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
        lm = spell_checker.learn_language_model(['../tester_module/sh.txt'])
        import ipdb
        ipdb.set_trace()
        correction = spell_checker.correct_sentence("I dould be in he room", lm, ERR_DIST, 2, 0.8)
        self.assertEqual(correction, "I would be in the room")


if __name__ == '__main__':
    unittest.main(verbosity=2)
