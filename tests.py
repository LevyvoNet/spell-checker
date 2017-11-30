import unittest
import spell_checker
import os

WORDS = spell_checker.get_word_counts(['data/corpora/big.txt'])
ERR_DIST = spell_checker.create_error_distribution('data/misspellings/wikipedia_common_misspellings.txt', WORDS)
lm = spell_checker.learn_language_model(['data/corpora/big.txt'])


def learn_stupid_text(func):
    text = """
    Hello, my name is Elad.
    I am happy to meet you.
    I love sea food and wine.
    """

    def new_func(*args, **kwargs):
        with open('temp_file.txt', 'wb') as f:
            f.write(text)

        test_lm = spell_checker.learn_language_model(['temp_file.txt'])
        args = args + (test_lm,)
        func(*args, **kwargs)
        os.remove('temp_file.txt')

    return new_func


class SpellCheckerTest(unittest.TestCase):
    @unittest.skip('just for now')
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
            print 'correcting {}...'.format(err)
            correction = spell_checker.correct_word(err, WORDS, ERR_DIST)
            print '{}->{}'.format(err, correction)
            self.assertEqual(correction, word,
                             'expected {}->{}, got {} instead'.format(err, word, correction))

    @unittest.skip('just for now')
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

    @unittest.skip('just for now')
    def test_correct_sentence(self):
        self.assertEqual(spell_checker.correct_sentence(
            "I would be in te room", lm, ERR_DIST, 1, 0.8),
            "I would be in the room")
        self.assertEqual(spell_checker.correct_sentence(
            "Go te the prison", lm, ERR_DIST, 1, 0.8),
            "Go to the prison")

    def test_correct_sentence_real_word(self):
        def test_correct_sentence(self):
            self.assertEqual(spell_checker.correct_sentence(
                "I would be in he room", lm, ERR_DIST, 2, 0.8),
                "I would be in the room")

    @unittest.skip('just for now')
    def test_get_n_from_language_model(self):
        lm_2 = spell_checker.learn_language_model(['data/corpora/big.txt'], 2, None)
        lm_3 = spell_checker.learn_language_model(['data/corpora/big.txt'], 3, None)
        lm_4 = spell_checker.learn_language_model(['data/corpora/big.txt'], 4, None)
        self.assertEqual(spell_checker.get_n_from_language_model(lm_2), 2)
        self.assertEqual(spell_checker.get_n_from_language_model(lm_3), 3)
        self.assertEqual(spell_checker.get_n_from_language_model(lm_4), 4)

    @unittest.skip('just for now')
    @learn_stupid_text
    def test_count_word_in_context(self, test_lm):
        self.assertEqual(spell_checker.get_counts_word_in_context('elad', ['is'], test_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('elad', ['name', 'is'], test_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('i', [''], test_lm), 2)
        self.assertEqual(spell_checker.get_counts_word_in_context('you', ['wanna', 'meet'], test_lm), 0)
        self.assertEqual(spell_checker.get_counts_word_in_context('you', ['meet'], test_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('elad', ['is'], test_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('elad', ['name', 'is'], test_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('', ['elad'], test_lm), 2)
        self.assertEqual(spell_checker.get_counts_word_in_context('', ['is', 'elad'], test_lm), 1)
        self.assertEqual(spell_checker.get_counts_word_in_context('love', ['', 'i'], test_lm), 1)

    @unittest.skip('just for now')
    @learn_stupid_text
    def test_get_counts_of_context(self, test_lm):
        self.assertEqual(spell_checker.get_counts_of_context(['my', 'name'], test_lm), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)
