import nltk

def Stemming(removed_stopwords_comments):
    stemmed_comments = []
    ps = nltk.PorterStemmer()
    for array in removed_stopwords_comments:
        temp = []
        for word in array:
            temp.append(list(ps.stem(i) for i in word))
        stemmed_comments.append(temp)
    return stemmed_comments