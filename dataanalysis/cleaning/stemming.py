from nltk.stem import PorterStemmer


def stemming(removed_stopwords_comments):
    """Performs stemming on a corpus of documents

    Takes a list of tokenized comments and removes the suffixes
    to make similar words equal such that they can be used more
    effectively and easier during vectorization andsemantic analysis.

    The stemming is performed with a PorterStemmer.

    Parameters
    ----------
    removed_stopwords_comments : list
        A list of comments where all stopwords are removed

    Note
    ----
    This function filters out and does not process comments with 3 or fewer tokens.

    Returns
    -------
    stemmed_comments : list
        A list of stemmed comments


    Examples
    --------
    >>> comments = [['running', 'quickly'], ['jumping', 'over', 'fence']]
    >>> stemming(comments)
    [['run', 'quickli'], ['jump', 'over', 'fence']]
    """
    print("Stemming started...")
    ps = PorterStemmer()

    #Cache to reduce the number of stemming operations and improve performance
    stem_cache = {}

    stemmed_comments = [
        [
            stem_cache[word]
            if word in stem_cache
            else stem_cache.setdefault(word, ps.stem(word))
            for word in comment_tokens
        ]
        for comment_tokens in removed_stopwords_comments
        if len(comment_tokens) > 3
    ]
    print("Pre-processing done...")
    return stemmed_comments
