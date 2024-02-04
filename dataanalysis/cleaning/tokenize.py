import regex as re
from nltk.tokenize import word_tokenize

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
    cleaned_comments = []
    tokenized_comments = []
    print("Pre-processing started...")
    print("Tokenizing...")
    for comment in comments:
        cleaned_comment = re.sub("[^-9A-Za-z ]", "", str(comment).strip())
        lowercased_comment = cleaned_comment.lower()
        cleaned_comments.append(lowercased_comment)
    for j in cleaned_comments:  
        tokenized_sentence = word_tokenize(j)
        tokenized_comments.append(tokenized_sentence)
    return tokenized_comments