import regex as re
from nltk.util import ngrams

''' Tokenizes the documents into bigrams

    Loops through every document and removes all punctuations
    and converts the text to lowercase. Both important for 
    further pre-processing, and to make the data as reliable 
    as possible
'''

def TokenizeComments(comments):
    tokenized = []

    for i in range(0, len(comments)):
        comments[i] = re.sub("[^-9A-Za-z ]", "", str(comments[i]).strip())
        comments[i] = comments[i].lower()

       #Tokenizing the comments into n-grams, with n=2, making it a bigram
        tokenized.append(list(ngrams(str(comments[i]).split(), 2)))
    return tokenized