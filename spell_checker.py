import itertools
import random
import re
import collections
from numpy import log

ALPHABET = 'abcdefghijklmnopqrstuvwxyz'
EMPTY_CHAR = '~'


class Edit(collections.namedtuple('Edit', ['error', 'type'], verbose=False)):
    def __str__(self):
        return '{}, {}'.format(self.error, self.type)

    def __repr__(self):
        return str(self)


class Distance(collections.namedtuple('Distance', ['price', 'edits'], verbose=False)):
    def __str__(self):
        return '({},{})'.format(self.price, self.edits)

    def __repr__(self):
        return str(self)


def prior_unigram(can, word_counts):
    """Calculate the prior of a word using a simple unigram language model.

    Use laplace smoothing for the calculation.
    Args:
        can(str): word.
        word_counts(dict): map from words to their counts in the language model.
    """
    return float(word_counts.get(can, 0) + 1) / (2 * len(word_counts.keys()))


def channel(edit, lexicon, errors_dist):
    """Return the channel part - the probability of a given edit.

    Args:
        edit(Edit): tuple represents an edit (e.g (('x','y'),'substitution).
        lexicon(dict): iterable structure contains all of the words in the language.
        errors_dist (dict): the errors distribution.

    Returns:
        float. the channel value for an edit.
    """
    default_channel_value = 1 / sum([d[err] for d in errors_dist.itervalues() for err in d])
    return errors_dist[edit.type].get(edit.error, default_channel_value)


def channel_multi_edits(edits, lexicon, errors_dist):
    """Calculate the channel of multiple edits under the assumption they are independent.

    Args:
        edits(list): list of tuples represents an edit (e.g (('x','y'),'substitution).
        lexicon(dict): iterable structure contains all of the words in the language.
        error_dist (dict): the errors distribution.

    Returns:
        float. the channel value for multiple edits.
    """
    return reduce(lambda x, y: x * y, [channel(edit, lexicon, errors_dist) for edit in edits], 1)


def get_counts_word_in_context(word, context, lm):
    """Return counts of word in a given context.

    Args:
        word(string): a word.
        context(list): the prefix of the word in a context (list of strings).
        lm(dict): the multigram language model.
    """
    if word not in lm:
        return 0

    total = 0
    context = ' '.join(context).strip().split(' ')
    for super_context in lm[word]:
        super_context_words = super_context.split(' ')
        valid = True
        if len(super_context_words) >= len(context):
            for i in range(len(context)):
                valid = valid and super_context_words[::-1][i] == context[::-1][i]

            if valid:
                total += lm[word][super_context]

    return total


def get_n_from_language_model(lm):
    return max(len(context.split(' ')) for context in lm['']) + 1


def unique_context_of_len_count(length, n, lm):
    if length == 0:
        return 1
    if length == 1:
        return len(lm.keys())

    n_grams = []
    for w in lm:
        for context in lm[w]:
            context_padded = (n - 1 - len(' '.join(context.split(' ')))) * [''] + context.split(' ')
            if len(context_padded) != n - 1:
                import ipdb
                ipdb.set_trace()
            n_grams.append(' '.join(context_padded[-(length - 1):] + [w]))

    return len(set(n_grams))


def get_counts_of_context(context, lm):
    context = ' '.join(context).strip().split(' ')
    total = 0
    for word in lm:
        for other_context in lm[word]:
            other_context_words = other_context.split(' ')
            valid = True
            if len(other_context_words) >= len(context):
                for i in range(len(context)):
                    valid = valid and other_context_words[::-1][i] == context[::-1][i]

                if valid:
                    total += lm[word][other_context]

    return total


def word_in_context_probability(word, context, n, lm):
    """Calculate the probability of a word appearing in a context.

    Args:
        word(string): a word.
        context(list): the prefix of the word in a context (list of strings).
            the length of context is n-1 for an n-gram model.
        n(int): n of the n-gram model.
        lm(dict): the multigram language model.

    Returns:
        float. the prior of the given word.
    """
    # TODO: what if word in not in lm? what if context is not in lm[word]?
    count_word_in_context = get_counts_word_in_context(word, context, lm)
    # number_of_contexts = unique_context_of_len_count(n - 1, n, lm)
    count_context = get_counts_word_in_context(context[-1], context[:-1], lm)
    number_of_contexts = 0
    if (count_context + number_of_contexts) == 0:
        return 0
    ret = float(count_word_in_context) / (count_context + number_of_contexts)
    return ret


def prior_multigram(word, context, n, lm):
    """Calculate the prior part using a multigram language model and backoff smoothing."""
    return word_in_context_probability(word, context, n, lm)

    #     sum += word_in_context_probability(word, context_padded, lm)
    # return sum([word_in_context_probability(word, context[i:], lm)
    #             for i in range(len(context))]) / n - 1


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

    Args:
        errors_file (str): full path to the errors file. File format mathces
                            Wikipedia errors list.
    Returns:
        dict. the count for each of the errors appear on the errors file.
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
           We ask you to keep it simple and straight forward for clarity.
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
            # smoothing for strings which do not appear in corpus.
            error_distribution[err_type][err] = \
                float(err_count + 1) / (string_count.get(original_string, 0) + 1)

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
        if edit_distance <= 2 and can != '':
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

    def candidate_score(can, edits):
        return log(prior_unigram(can, word_counts)) + \
               log(channel_multi_edits(edits, word_counts, errors_dist))

    candidates_scores = {
        can: candidate_score(can, errors)
        for can, errors in itertools.izip(*generate_candidates(w, word_counts.iterkeys()))
    }
    return max(candidates_scores.iterkeys(), key=lambda c: candidates_scores[c])


def normalize_text(text):
    """Return text in a normalized form."""
    text = text.lower()
    chars_to_remove = ['\n', '\r', '\t']
    chars_to_convert_to_word = ['!', '?', '-', '_']
    text_after_remove_chars = reduce(lambda s, char: s.replace(char, ' '),
                                     chars_to_remove, text)

    return reduce(lambda s, char: s.replace(char, ' ' + char),
                  chars_to_convert_to_word, text_after_remove_chars)


def extract_sentences(text):
    """Extract sentences from a text.

    Args:
        text(string).

    Returns:
        list. a list of strings represents sentences appeared in the given text.
    """
    chars_to_remove = [',', '"', '(', ')', '_', '-']
    return [
        reduce(lambda s, char: s.replace(char, ''), chars_to_remove, s).lstrip()
        for s in re.split('\.', text)
    ]


def get_word_counts(files):
    """Return a simple unigram language model which just count word appearances.

    Args:
        files(list): a list of files path which contains texts.

    Returns:
        dict. histogram of word counts.
    """
    ret = {}
    for file in files:
        with open(file) as f:
            ret.update(collections.Counter(re.findall(r'\w+', f.read().lower())))

    return ret


def add_ngram_to_ngrams_count(ngram, ngrams_count):
    """Add ngram to a given ngrams count.

    Args:
        ngram(list): a list of words represents a sentence.
        ngrams_count(dict): count of ngrams.
    """
    word = ngram[-1]
    context = ' '.join(ngram[:-1]).strip()
    if word not in ngrams_count:
        ngrams_count[word] = dict()

    if word == '' and context == '':
        # This is meaning less and just noise.
        return

    ngrams_count[word][context] = ngrams_count[word].get(context, 0) + 1


def generate_ngrams_from_sentence(sentence, n):
    """Generate ngrams of length n from sentence.

    Args:
        sentence(str): sentence.
        n(int): the length of each ngram.
    """
    sentence_words = sentence.split(' ')
    # Pad the sentence with empty strings in it's begining and ending.
    sentence_words = [''] * (n - 1) + sentence_words + [''] * (n - 1)
    return [sentence_words[i:i + n]
            for i in range(len(sentence_words) - n + 1)]


def learn_language_model(files, n=3, lm=None):
    """Return a language model based on the specified files.

    Text normalization includes:
        1. Removing all white spaces (as well as line breaks, tabs, etc.) becuase sometimes
            sentences are splitted between several lines.
        2. Convert all of the text to lower case for treating the word itself
            ('food' and 'Food' is considered the same word).
        3. Remove ',' and '"' character from the text (the simplest way to remove noise
            which causes 'hello,' and 'hello' considered differently).
        4. Make '!','?' characters independent words by adding a space before each instance of them.

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
    'w1': {'': 1, 'w2 w3': 1},
    'w2': {'w1': 1},
    'w3': {'w1 w2': 1},
    'w4': {'w3 w1': 1},
    '': {'w1 w4': 1, 'w4': 1}
    }

    Args:
        files (list): a list of files (full path) to process.
        n (int): length of ngram, default 3.
        lm (dict): update and return this dictionary if not None.
                   (default None).

    Returns:
        dict: a nested dict {str:{str:int}} of ngrams and their counts.
    """
    if n == 1:
        return get_word_counts(files)

    language_model = {}
    for file in files:
        with open(file) as f:
            sentences = extract_sentences(normalize_text(f.read()))
            for sentence in sentences:
                for ngram in generate_ngrams_from_sentence(sentence, n):
                    add_ngram_to_ngrams_count(ngram, language_model)

    if lm is not None:
        lm.update(language_model)

    return language_model


def capitalize_if_needed(new_sentence, original_sentence):
    """Return the new sentence with capital letters where needed.

    When correcting a sentence a normalization which includes
    text lowering is executed. Because of that the corrected sentence
    might miss capital letters, for example: 'I want to eat dinnar' -> 'i want to eat dinner'.
    The purpose of this function is to convert it to 'I want to eat dinner'
    """
    new_sentence_words = new_sentence.split(' ')
    original_sentence_words = original_sentence.split(' ')
    for i in range(min(len(new_sentence_words), len(original_sentence_words))):
        if new_sentence_words[i].lower() == original_sentence_words[i].lower():
            new_sentence_words[i] = original_sentence_words[i]

    return ' '.join(new_sentence_words)


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
    s = normalize_text(s)
    s = [''] * (n - 1) + s.split(' ') + [''] * (n - 1)
    log_likelihood = 0
    for i in range(len(s) - 2 * (n - 1)):
        word = s[i + n - 1]
        context = s[i:i + n - 1]
        # print 'word: {}, context: {}, score:{}'.format(word, context, prior_multigram(word, context, lm))
        log_likelihood += prior_multigram(word, context, n, lm)

    return log_likelihood


def weighted_choice(choices):
    total = sum(choices[c] for c in choices.iterkeys())
    r = random.uniform(0, total)
    upto = 0
    for c in choices:
        if upto + choices[c] >= r:
            return c
        upto += choices[c]
    assert False, "Shouldn't get here"


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
    # TODO: implement this faster and probabilistic
    text_words = ['' if w == None else normalize_text(w)]
    while len([word for word in text_words if word != '']) < m:
        # TODO: what if possible_words is empty? what about the empty word?
        possible_words = lm.iterkeys()
        n = get_n_from_language_model(lm)
        if len(text_words) < n - 1:
            text_last_words = text_words
        else:
            text_last_words = text_words[-(n - 1):]
        possible_words_scores = {word: prior_multigram(word, text_last_words, n, lm)
                                 for word in possible_words}

        chosen_word = weighted_choice(possible_words_scores)
        text_words.append(chosen_word)
        print 'current text is: {}'.format(' '.join(text_words))

    return ' '.join([word for word in text_words if word != ''])


def correct_word_in_sentence(sentence_words, lm, err_dist, word_index, alpha):
    """Correct specific word in a sentence.

    Args:
        sentence_words (list): the sentence to correct (list of strings).
        lm (dict): the language model to correct the sentence accordingly.
        err_dist (dict): error distributions according to error types
                        (as returned by create_error_distribution() ).
        word_index (int): the word index to correct
        alpha (float): the likelihood of a lexical entry to be the a correct word.
                        (default: 0.95)

    Returns:
        str. the sentence with the specified word replaced by the best one
            according to the given errors distributions and language model.
    """

    lexicon = lm.keys()
    context = sentence_words[:word_index]
    n = get_n_from_language_model(lm)
    if len(context) < n - 1:
        context_last_words = context
    else:
        context_last_words = context[-(n - 1):]

    if context_last_words == []:
        import ipdb
        ipdb.set_trace()

    def channel_for_sentence(original_word, can, edits, alpha):
        if original_word == can:
            return alpha

        return channel_multi_edits(edits, lexicon, err_dist)

    def candidate_score(original_word, can, edits):
        return log(prior_multigram(can, context_last_words, n, lm)) + \
               log(channel_for_sentence(original_word, can, edits, alpha))

    word_to_correct = sentence_words[word_index]
    candidates = generate_candidates(word_to_correct, lexicon)

    candidates_scores = {
        can: candidate_score(word_to_correct, can, edits)
        for can, edits in itertools.izip(*candidates)
    }

    new_word = max(candidates_scores.iterkeys(), key=lambda c: candidates_scores[c])
    new_sentence_words = context + [new_word] + sentence_words[word_index + 1:]
    return ' '.join(new_sentence_words).strip()


def correct_multiple_words_in_sentence(sentence_words, lm, err_dist, word_indices, alpha):
    """Correct specific word in a sentence.

    Note: the current implementation is not prune to cases where it is the best to switch
        multiple words on the 'same time' since it corrects word after word and not multiple
        words at once.

    Args:
        sentence_words (list): the sentence to correct (list of strings).
        lm (dict): the language model to correct the sentence accordingly.
        err_dist (dict): error distributions according to error types
                        (as returned by create_error_distribution() ).
        word_indices (list): list of word indices to correct
        alpha (float): the likelihood of a lexical entry to be the a correct word.
                        (default: 0.95)

    Returns:
        str. the sentence with the specified words replaced by the best one
            according to the given errors distributions and language model.
    """
    for word_index in word_indices:
        sen = correct_word_in_sentence(sentence_words, lm, err_dist, word_index, alpha)
        sen = normalize_text(sen)
        sentence_words = [''] + sen.split(' ')

    return sen
    # return reduce(lambda sen, i:
    #               correct_word_in_sentence(sen, lm, err_dist, i, alpha), word_indices,
    #               sentence_words)


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
    # TODO: cache sentence,index combinations to improve run time.
    # TODO: prefer fixing non-words before real words.
    n = get_n_from_language_model(lm)
    orig_s = s
    s = normalize_text(s)
    # This is a sentence an empty string at the start is required.
    sentence_words = [''] + s.split(' ')
    # indices_not_in_language = [i for i in len(sentence_words) if sentence_words[i] not in lm.keys()]
    indices_combinations = itertools.combinations(range(1, len(sentence_words)), c)
    candidate_sentences_scores = {}
    for indices in indices_combinations:
        new_sentence = correct_multiple_words_in_sentence(
            sentence_words, lm, err_dist, indices, alpha)

        candidate_sentences_scores[new_sentence] = evaluate_text(new_sentence, n, lm)
        print 'suggested sentence is: {}, score: {}'.format(new_sentence,
                                                            candidate_sentences_scores[new_sentence])

    final_sentence = max(candidate_sentences_scores.iterkeys(),
                         key=lambda can: candidate_sentences_scores[can])

    return capitalize_if_needed(final_sentence, orig_s)
    # TODO: remove all print, ipdb and commented out code
