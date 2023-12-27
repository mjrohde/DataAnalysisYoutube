import nltk

''' Removes stopwords from list of comments

    Uses the nltk's corpus of stopwords to eliminate stopwords
    from each document in the corpus. The function uses a loop
    to iterate over each word in every document, and removes
    the word if it exist in nltk's corpus. If not contained
    it is entered into a new list.

    All empty arrays are removed from the newly created list as
    a result of the removal  
'''

def RemoveStopwords(tokenized_comments):
    new_comments = []
    stopwords = nltk.corpus.stopwords.words('english')
    for i in tokenized_comments:
        temp = []
        stopped = False
        for j in i:
            stopped = False
            for k in j:
                if k in stopwords:
                    stopped = True
            if stopped == False:
                temp.append(j)
        new_comments.append(temp)

    #Remove all empty arrays as a result of the stopwords removal
    new_comments = [ele for ele in new_comments if ele != []]
    print(new_comments)
    return new_comments