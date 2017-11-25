import itertools
from collections import namedtuple, defaultdict
from numpy import log

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
EMPTY_CHAR = '~'


class Edit(namedtuple('Edit', ['error', 'type'], verbose=False)):
    def __str__(self):
        return '{}, {}'.format(self.error, self.type)

    def __repr__(self):
        return str(self)


class Distance(namedtuple('Distance', ['price', 'edits'], verbose=False)):
    def __str__(self):
        return '({},{})'.format(self.price, self.edits)

    def __repr__(self):
        return str(self)


def learn_language_model(files, n=3, lm=None):
    """ Returns a nested dictionary of the language model based on the
    specified files.Text normalization is expected (please explain your choice
    of normalization HERE in the function docstring.
    Example of the returned dictionary for the text 'w1 w2 w3 w1 w4' with a
    tri-gram model:
    tri-grams:
    <> <> w1
    <> w1 w2
    w1 w2 w3
    w2 w3 w1
    w3 w1 w4
    w1 w4 <>
    w4 <> <>
    and returned language model is:
    {
    w1: {'':1, 'w2 w3':1},
    w2: {w1:1},
    w3: {'w1 w2':1},
    w4:{'w3 w1':1},
    '': {'w1 w4':1, 'w4':1}
    }

    Args:
     	  files (list): a list of files (full path) to process.
          n (int): length of ngram, default 3.
          lm (dict): update and return this dictionary if not None.
                     (default None).

    Returns:
        dict: a nested dict {str:{str:int}} of ngrams and their counts.
    """

    return ngrams


def get_string_count(word_count):
    """Return count of strings at the size of 1,2 in word_count
    
    We are going to use this in order to create the error distribution (this is the denominator).

    Args:
        word_count(dict): A dictionary of words and their counts.

    Returns:
        dict. the matrix of strings at the size of 1,2 and their counts in the given dict.
            for example: {'ab': 5, 'x': 1...}
    """
    ret = {}
    for word in word_count:
        for i in range(len(word) - 2):
            ret[word[i] + word[i + 1]] = ret.get(word[i] + word[i + 1], 0) + word_count[word]
            ret[word[i]] = ret.get(word[i], 0) + word_count[word]
            # for the last character of word.

        ret[word[-1]] = ret.get(word[-1], 0) + word_count[word]

    ret[EMPTY_CHAR] = len(word_count)
    return ret


def get_error_count(errors_file):
    """Return the count of errors occurred in the given file.

    The structure is the same as the confusion matrix representation. Means {err_type: dict} err_type
    is in ['deletion', 'insertion', 'substitution', 'transposition'] and dict is {(corr,err): count}
    """
    # Smooth the error count using laplace correction.
    error_count = {'deletion': {(x + y, x): 1 for x, y in itertools.permutations(ALPHABET, 2)},
                   'insertion': {(x, x + y): 1 for x, y in itertools.permutations(ALPHABET, 2)},
                   'substitution': {(x, y): 1 for x, y in itertools.permutations(ALPHABET, 2)},
                   'transposition': {(x + y, y + x): 1 for x, y in itertools.permutations(ALPHABET, 2)}
                   }
    error_count['deletion'].update({(x, EMPTY_CHAR): 1 for x in ALPHABET})
    error_count['deletion'].update({(x + x, x): 1 for x in ALPHABET})
    error_count['insertion'].update({(EMPTY_CHAR, x): 1 for x in ALPHABET})
    error_count['insertion'].update({(x, x + x): 1 for x in ALPHABET})

    with open(errors_file, 'r') as misspellings:
        for line in misspellings:
            # cut \n
            [misspelled_words, real_word] = line.lower()[:-1].split('->')
            misspelled_words = misspelled_words.split(', ')
            for misspelled_word in misspelled_words:
                _, edits = optimal_string_alignment(real_word, misspelled_word)
            errors = [edit for edit in edits if edit.error[0] != edit.error[1]]
            for err in errors:
                error_count[err.type][err.error] = error_count[err.type].get(err.error, 0) + 1

    return error_count


def create_error_distribution(errors_file, lexicon):
    """ Returns a dictionary {str:dict} where str is in:
    <'deletion', 'insertion', 'transposition', 'substitution'> and the inner dict {tuple: float} represents the confution matrix of the specific errors
    where tuple is (err, corr) and the float is the probability of such an error. Examples of such tuples are ('t', 's'), ('-', 't') and ('ac','ca').
    Notes:
        1. The error distributions could be represented in more efficient ways.
           We ask you to keep it simpel and straight forward for clarity.
        2. Ultimately, one can use only 'deletion' and 'insertion' and have
            'sunstiturion' and 'transposition' derived. Again,  we use all
            four explicitly in order to keep things simple.
    Args:
        errors_file (str): full path to the errors file. File format mathces
                            Wikipedia errors list.
        lexicon (dict): A dictionary of words and their counts derived from
                        the same corpus used to learn the language model.

    Returns:
        A dictionary of error distributions by error type (dict).

    """
    # initialize the error model with a laplace smoothing.
    error_distribution = {'deletion': dict(),
                          'insertion': dict(),
                          'substitution': dict(),
                          'transposition': dict()}

    errors_count = get_error_count(errors_file)
    string_count = get_string_count(lexicon)

    for err_type, type_count in errors_count.iteritems():
        for err, err_count in type_count.iteritems():
            original_string = err[0]
            # Here we are using laplace smoothing for string appearances.
            error_distribution[err_type][err] = \
                float(err_count) / (string_count.get(original_string, 0) + 1)

    return error_distribution


def optimal_string_alignment(src, dst):
    """Calculate OSA distance between two strings and the edit series from one to the other.

    Args:
        src(string): a word.
        dst(string): another word.

    Returns:
        Distance. (price, edits) while price is the edit distance between the two
            strings (number) and edits is a list contains Edit represents the
            edits from w to s.
    """

    def nice_print_d(d):
        two_dim_arr = [[d[(i, j)] for i in range(len(src))] for j in range(len(dst))]
        for i in range(1, len(two_dim_arr)):
            print two_dim_arr[i][1:]

    # arbitrary value to initialize the dynamic programming matrix, no cell should be left with this value
    # in the end.
    init_val = -100
    src, dst = EMPTY_CHAR + src, EMPTY_CHAR + dst

    # Initialize a dictionary which represents the dynamic programming matrix.
    d = {(i, j): Distance(init_val, [])
         for i in range(len(src))
         for j in range(len(dst))}

    # Initialize the first column of d with deletions.
    for i in range(len(src)):
        edit = d[i - 1, 0].edits + [Edit((src[i], EMPTY_CHAR), 'deletion')] if i != 0 else []
        d[i, 0] = Distance(i, edit)

    # Initialize the first row of d with insertions.
    for j in range(len(dst)):
        edit = d[0, j - 1].edits + [Edit((EMPTY_CHAR, dst[j]), 'insertion')] if j != 0 else []
        d[0, j] = Distance(j, edit)

    for i in range(len(src))[1:]:
        for j in range(len(dst))[1:]:
            substitution_cost = 0 if src[i] == dst[j] else 1

            deletion = Edit((src[i - 1] + src[i], src[i - 1]), 'deletion')
            insertion = Edit((dst[j - 1], dst[j - 1] + dst[j]), 'insertion')
            substitution = Edit((src[i], dst[j]), 'substitution')

            possible_edits = {
                deletion: Distance(d[i - 1, j].price + 1,
                                   d[i - 1, j].edits + [deletion]),
                insertion: Distance(d[i, j - 1].price + 1,
                                    d[i, j - 1].edits + [insertion]),
                substitution: Distance(d[i - 1, j - 1].price + substitution_cost,
                                       d[i - 1, j - 1].edits + [substitution])
            }

            if i > 1 and j > 1 and src[i] == dst[j - 1] and src[i - 1] == dst[j]:
                transposition = Edit((src[i - 1] + src[i], dst[j - 1] + dst[j]), 'transposition')
                possible_edits[transposition] = Distance(d[i - 2, j - 2].price + 1,
                                                         d[i - 2, j - 2].edits + [transposition])

            best_edit = min(possible_edits.iterkeys(), key=lambda e: possible_edits[e].price)
            d[i, j] = possible_edits[best_edit]

    return d[(len(src) - 1, len(dst) - 1)]


def generate_candidates(misspelled_word, lexicon):
    """Generate candidates for a word correction and their matching edits.

    Args:
        misspelled_word(string): the misspelled word.
        lexicon(list): list contains all of the words.

    Returns:
        tuple. (candidates, edits) tuples contains the
            candidate words (list) and their edits from given misspelled word w
            (list of tuples represents edits).
    """
    candidates, errors = [], []
    for can in lexicon:
        edit_distance, edits = optimal_string_alignment(can, misspelled_word)
        if edit_distance <= 2:
            candidates.append(can)
            # avoid appending "empty" substitutions (e.g ('a', 'a'))
            real_errors = [edit for edit in edits if edit.error[0] != edit.error[1]]
            errors.append(real_errors)

    return candidates, errors


def correct_word(w, word_counts, errors_dist):
    """ Returns the most probable correction for the specified word, given the specified prior error distribution.

    Args:
        w (str): a word to correct
        word_counts (dict): a dictionary of {str:count} containing the
                            counts  of uniqie words (from previously loaded
                             corpora).
        errors_dist (dict): a dictionary of {str:dict} representing the error
                            distribution of each error type (as returned by
                            create_error_distribution() ).

    Returns:
        The most probable correction (str).
    """
    default_channel_value = 1 / len(word_counts)

    def prior(can):
        return float(word_counts[can]) / len(word_counts.keys())

    def channel(edit):
        return errors_dist[edit.type].get(edit.error, default_channel_value)

    def channel_multi_edits(edits):
        return reduce(lambda x, y: x * y, [channel(edit) for edit in edits], 1)

    def candidate_score(can, edits):
        return log(prior(can)) + log(channel_multi_edits(edits))

    candidates_score = {
        can: candidate_score(can, edits)
        for can, edits in itertools.izip(*generate_candidates(w, word_counts.iterkeys()))
    }
    print candidates_score

    return max(candidates_score.iterkeys(), key=lambda c: candidates_score[c])


def generate_text(lm, m=15, w=None):
    """ Returns a text of the specified length, generated according to the
     specified language model using the specified word (if given) as an anchor.

     Args:
        lm (dict): language model used to generate the text.
        m (int): length (num of words) of the text to generate (default 15).
        w (str): a word to start the text with (default None)

    Returns:
        A sequrnce of generated tokens, separated by white spaces (str)
    """


def correct_sentence(s, lm, err_dist, c=2, alpha=0.95):
    """ Returns the most probable sentence given the specified sentence, language
    model, error distributions, maximal number of suumed erroneous tokens and likelihood for non-error.

    Args:
        s (str): the sentence to correct.
        lm (dict): the language model to correct the sentence accordingly.
        err_dist (dict): error distributions according to error types
                        (as returned by create_error_distribution() ).
        c (int): the maximal number of tokens to change in the specified sentence.
                 (default: 2)
        alpha (float): the likelihood of a lexical entry to be the a correct word.
                        (default: 0.95)

    Returns:
        The most probable sentence (str)

    """


def evaluate_text(s, n, lm):
    """ Returns the likelihood of the specified sentence to be generated by the
    the specified language model.

    Args:
        s (str): the sentence to evaluate.
        n (int): the length of the n-grams to consider in the language model.
        lm (dict): the language model to evaluate the sentence by.

    Returns:
        The likelihood of the sentence according to the language model (float).
    """
