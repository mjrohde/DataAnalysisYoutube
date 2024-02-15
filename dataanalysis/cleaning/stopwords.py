import nltk
import sys

def remove_stopwords(tokenized_comments):
    ''' Removes stopwords from list of comments

    Combines a custom stopwords list with the nltk's corpus of stopwords to eliminate stopwords
    from each document in the corpus. The function uses list comprehension 
    to check if a token is in the set of stopwords, and adds the words not contained
    in the set to a list.

    Parameters
    ----------
    tokenized_comments : list 
        A list of tokenized comments

    Returns
    -------
    filtered_comments : list
        A list of comments with all stopwords removed  
    '''
    print('Removing stopwords...')
    stopwords_list = nltk.corpus.stopwords.words('english')

    with open(f'{sys.path[0]}/cleaning/stopwords.txt', 'r') as f:
        custom_stopwords = {line.strip().lower() for line in f}
    stopwords_set = set(stopwords_list).union(custom_stopwords)

    filtered_comments = [
        [token.strip() for token in sentence_tokens if token.strip() not in stopwords_set]
        for sentence_tokens in tokenized_comments if sentence_tokens
    ]

    return filtered_comments