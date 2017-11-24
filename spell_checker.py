import itertools
import functools
from collections import namedtuple
from numpy import log

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'


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
    with open(errors_file, 'r') as erros:
        for err in erros:
            [misspelled_word, real_word] = err.split('->')
            print 'misspelled:{}, real: {}'.format(misspelled_word, real_word)


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


def optimal_string_alignment(w, s):
    """Calculate OSA distance between two strings and the edit series from one to the other.

    Args:
        w(string): a word.
        s(string): another word.

    Returns:
        Distance. (price, edits) while price is the edit distance between the two
            strings (number) and edits is a list contains Edit represents the
            edits from w to s.
    """
    # def nice_print_d(d):
    #     two_dim_arr = [[d[(i, j)] for i in range(len(w) + 1)] for j in range(len(s) + 1)]
    #     for i in range(1, len(two_dim_arr)):
    #         print two_dim_arr[i][1:]

    # arbitrary value to initialize the dynamic programming matrix, no cell should be left with this value
    # in the end.
    init_val = -100

    # Initialize a dictionary which represents the dynamic programming matrix.
    d = {(i, j): Distance(init_val, [])
         for i in range(-1, len(w) + 1)
         for j in range(-1, len(s) + 1)}

    # Initialize the first column of d with deletions.
    for i in range(len(w) + 1):
        edit = d[i - 1, 0].edits + [Edit((w[i - 1], '-'), 'deletion')] if i != 0 else []
        d[i, 0] = Distance(i, edit)

    # Initialize the first row of d with insertions.
    for j in range(len(s) + 1):
        edit = d[0, j - 1].edits + [Edit(('-', s[j - 1]), 'insertion')] if j != 0 else []
        d[0, j] = Distance(j, edit)

    for i in range(len(w) + 1)[1:]:
        for j in range(len(s) + 1)[1:]:
            substitution_cost = 0 if w[i - 1] == s[j - 1] else 1

            deletion = Edit((w[i - 1], '-'), 'deletion')
            insertion = Edit(('-', s[j - 1]), 'insertion')
            substitution = Edit((w[i - 1], s[j - 1]), 'substitution')

            possible_edits = {
                deletion: Distance(d[i - 1, j].price + 1,
                                   d[i - 1, j].edits + [deletion]),
                insertion: Distance(d[i, j - 1].price + 1,
                                    d[i, j - 1].edits + [insertion]),
                substitution: Distance(d[i - 1, j - 1].price + substitution_cost,
                                       d[i - 1, j - 1].edits + [substitution])
            }

            if i > 1 and j > 1 and w[i - 1] == s[j - 2] and w[i - 2] == s[j - 1]:
                transposition = Edit((w[i - 2] + w[i - 1], s[j - 2] + s[j - 1]), 'transposition')
                possible_edits[transposition] = Distance(d[i - 2, j - 2].price + 1,
                                                         d[i - 2, j - 2].edits + [transposition])

            best_edit = min(possible_edits.iterkeys(), key=lambda e: possible_edits[e].price)
            d[i, j] = possible_edits[best_edit]

    return d[(len(w), len(s))]


# def damerau_levenshtein_and_edits(w, s):
#     """Calculate Damerau-Levenshtein distance between two strings edit series from one to the other.
#
#     Args:
#         w(string): a word.
#         s(string): another word.
#
#     Returns:
#         tuple. (distance, edits) while distance is the edit distance between the two
#             strings (number) and edits is a list contains Edit represents the
#             edits from s to w.
#     """
#
#     def nice_print_d(d):
#         two_dim_arr = [[d[(i, j)] for i in range(len(w) + 1)] for j in range(len(s) + 1)]
#         # two_dim_arr = [[i - 1 for i in range(len(w) + 1)]] + two_dim_arr
#         # for i in range(len(s) + 1):
#         #     two_dim_arr[i] = [i - 1] + two_dim_arr[i]
#
#         for i in range(1, len(two_dim_arr)):
#             print two_dim_arr[i][1:]
#
#     init_val = -100
#     max_dist = len(w) + len(s)
#     da = {x: 0 for x in ALPHABET}
#
#     # initialize the dynamic programming matrix.
#     d = {(i, j): Distance(init_val, [])
#          for i in range(-1, len(w) + 1)
#          for j in range(-1, len(s) + 1)}
#
#     d[(-1, -1)] = Distance(max_dist, [])
#
#     for i in range(len(w) + 1):
#         d[(i, -1)] = Distance(max_dist, [])
#         edit = d[i - 1, 0].edits + [Edit(('-', w[i - 1]), 'deletion')] if i != 0 else []
#         d[(i, 0)] = Distance(i, edit)
#
#     for j in range(len(s) + 1):
#         d[(-1, j)] = Distance(max_dist, [])
#         edit = d[0, j - 1].edits + [Edit((s[j - 1], '-'), 'insertion')] if j != 0 else []
#         d[(0, j)] = Distance(j, edit)
#
#     for i in range(len(w) + 1)[1:]:
#         db = 0
#         for j in range(len(s) + 1)[1:]:
#             k = da[s[j - 1]]
#             l = db
#             if w[i - 1] == s[j - 1]:
#                 cost = 0
#                 db = j
#             else:
#                 cost = 1
#
#             possible_edits = {
#                 Edit(('-', w[i - 1]), 'deletion'): Distance(d[i - 1, j].price + 1,
#                                                             d[i - 1, j].edits + [Edit(('-', w[i - 1]), 'deletion')]),
#                 Edit((s[j - 1], '-'), 'insertion'): Distance(d[i, j - 1].price + 1,
#                                                              d[i, j - 1].edits + [Edit((s[j - 1], '-'), 'insertion')]),
#                 Edit((s[j - 1], w[i - 1]), 'substitution'): Distance(d[i - 1, j - 1].price + cost,
#                                                                      d[i - 1, j - 1].edits + [
#                                                                          Edit((s[j - 1], w[i - 1]), 'substitution')]),
#                 Edit((s[j - 2] + s[j - 1], w[i - 2] + w[i - 1]), 'transposition'):
#                     Distance(d[k - 1, l - 1].price + (i - k - 1) + 1 + (j - l - 1),
#                              d[i - 2, j - 2].edits + [
#                                  Edit((s[j - 2] + s[j - 1], w[i - 2] + w[i - 1]),
#                                       'transposition')])
#             }
#             best_edit = min(possible_edits.iterkeys(), key=lambda e: possible_edits[e].price)
#             d[(i, j)] = possible_edits[best_edit]
#
#             da[w[i - 1]] = i
#
#     nice_print_d(d)
#     return d[(len(w), len(s))]


def generate_candidates(w, d):
    """Generate candidates for a word correction and their matching edits.

    Args:
        w(string): the misspelled word.
        d(list): dictionary contains all of the words.

    Returns:
        tuple. (candidates, edits) tuples contains the
            candidate words (list) and their edits from given misspelled word w
            (list of tuples represents edits).
    """
    candidates, edits = [], []
    for can in d:
        edit_distance, edits = optimal_string_alignment(can, w)
        if edit_distance <= 2:
            candidates.append(can)
            edits.append(edits)

    return candidates, edits


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

    def prior(can):
        return float(word_counts[can]) / len(word_counts.keys())

    def channel(edit):
        return errors_dist[edit.type][edit.error]

    def channel_multi_edits(edits):
        return reduce(lambda x, y: x * y, [channel(edit) for edit in edits], 1)

    def candidate_prob(can, edits):
        return log(prior(can)) + log(channel_multi_edits(edits))

    can_probs = {
        can: candidate_prob(can, edits)
        for can, edits in itertools.izip(generate_candidates(w, word_counts.keys()))
    }
    return max(can_probs.iterkeys(), key=lambda c: can_probs[c])


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
