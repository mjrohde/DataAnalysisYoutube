import regex as re
from nltk.tokenize import word_tokenize

def tokenize_comments(comments):
    ''' Tokenizes the documents for further processing

    First, it removes all punctuation and symbols from each comment, then converts the text to lowercase. 
    These preprocessing steps are important for normalizing the text for further analysis.

    Parameters
    ----------
    comments : list
        A list of all comments extracted from the csv

    Returns
    -------
    tokenized_comments : list     
        A list of lowercase, no punctuation or symbols tokens
    '''
    tokenized_comments = []
    print('Pre-processing started...')
    print('Tokenizing...')
    for comment in comments:
        cleaned_comment = re.sub('[^0-9A-Za-z ]', '', str(comment).strip())
        lowercased_comment = cleaned_comment.lower()
        tokenized_comments.append(word_tokenize(lowercased_comment))
    return tokenized_comments