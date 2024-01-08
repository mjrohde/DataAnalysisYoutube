import regex as re
from nltk.util import ngrams

def tokenize_comments(comments):
    ''' Tokenizes the documents into bigrams

    Loops through every document and removes all punctuations
    and converts the text to lowercase. Both important for 
    further pre-processing, and to make the data as reliable 
    as possible

    Parameters
    ----------
    comments : list
        A list of all comments extracted from the csv

    Returns
    -------
    ngram_tokens : list     
        A list of lowercase, no punctuation or symbols bigrams
    '''
    ngram_tokens = []
    for comment in comments:
        cleaned_comment = re.sub("[^-9A-Za-z ]", "", str(comment).strip())
        lowercased_comment = cleaned_comment.lower()

        ngram_tokens.append(list(ngrams(str(lowercased_comment).split(), 2)))
    return ngram_tokens