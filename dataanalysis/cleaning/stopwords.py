import nltk

def RemoveStopwords(tokenized_comments):
    new_comments = []
    stopwords = nltk.corpus.stopwords.words('english')
    index = 0
    for i in tokenized_comments:
        index += 1
        temp = []
        for j in i:
            temp.append(list(i for i in j if i not in stopwords))
        new_comments.append(temp)

    #Remove all empty arrays as a result of the stopwords removal
    new_comments = [[ele for ele in array if ele != []] for array in new_comments]

    return new_comments