import nltk

def remove_stopwords(tokenized_comments):
    ''' Removes stopwords from list of comments

    Uses the nltk's corpus of stopwords to eliminate stopwords
    from each document in the corpus. The function uses a loop
    to iterate over each word in every document, and removes
    the word if it exist in nltk's corpus. If not contained
    it is entered into a new list.

    All empty arrays are removed from the newly created list as
    a result of the removal

    Parameters
    ----------
    tokenized_comments : A list of tokenized comments

    Returns
    -------
    filtered_comments : A list of comments with all stopwords removed  
    '''
    filtered_comments = []
    stopwords_list = nltk.corpus.stopwords.words('english')
    for sentence_tokens in tokenized_comments:
        filtered_words = []
        stop_detected = False
        for word in sentence_tokens:
            stop_detected = any(token in stopwords_list for token in word)
            if not stop_detected:
                filtered_words.append(word)
        filtered_comments.append(filtered_words)

    #Remove all empty arrays as a result of the stopwords removal
    filtered_comments = [sentence for sentence in filtered_comments if sentence != []]
    return filtered_comments