import spell_checker
import time


def main():
    WORDS = spell_checker.get_word_counts(['data/corpora/big.txt'])
    ERR_DIST = spell_checker.create_error_distribution('data/misspellings/wikipedia_common_misspellings.txt', WORDS)
    start = time.time()
    misspelled_word = 'beter'
    corrected_word = spell_checker.correct_word(misspelled_word, WORDS, ERR_DIST)
    end = time.time()
    print '{}->{}'.format(misspelled_word, corrected_word)
    print 'it took {} seconds'.format(end - start)


if __name__ == '__main__':
    main()
