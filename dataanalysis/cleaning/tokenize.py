import regex as re
from nltk.util import ngrams

def TokenizeComments(comments):
    tokenized = []

    for i in range(0, len(comments)):
        comments[i] = re.sub("[^-9A-Za-z ]", "", str(comments[i]).strip())
        comments[i] = comments[i].lower()

       #Tokenizing the comments into n-grams, with n=2, making it a bigram
        tokenized.append(list(ngrams(str(comments[i]).split(), 2)))
    return tokenized