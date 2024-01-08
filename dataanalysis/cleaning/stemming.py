import nltk

def stemming(removed_stopwords_comments):
    ''' Performs stemming on a corpus of documents

    Takes a list of ngrams where stopwords are removed based
    on nltk's corpus of stopwords in english. The stemming is performed
    with a PorterStemmer.

    Parameters
    ----------
    removed_stopwords_comments : list
        A list of comments where all stopwords are removed

    Returns
    -------
    stemmed_comments : list
        A list of stemmed comments


    Examples
    --------
    >>> comments = [['running', 'quickly'], ['jumping', 'over', 'fence']]
    >>> stemming(comments)
    [['run', 'quickli'], ['jump', 'over', 'fence']]
    '''
    stemmed_comments = []
    ps = nltk.PorterStemmer()
    for array in removed_stopwords_comments:
        temp = []
        for word in array:
            temp.append(list(ps.stem(i) for i in word))
        stemmed_comments.append(temp)
    return stemmed_comments