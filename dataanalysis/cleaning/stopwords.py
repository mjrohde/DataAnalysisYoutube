import nltk

def remove_stopwords(tokenized_comments):
    ''' Removes stopwords from list of comments

    Uses the nltk's corpus of stopwords to eliminate stopwords
    from each document in the corpus. The function uses a loop
    to iterate over each word in every document, and removes
    the word if it exist in nltk's corpus. If not contained
    it is entered into a new list.

    After the removal process, any resulting empty arrays are filtered out.

    Parameters
    ----------
    tokenized_comments : list 
        A list of tokenized comments

    Returns
    -------
    filtered_comments : list
        A list of comments with all stopwords removed  
    '''
    filtered_comments = []
    stopwords_list = nltk.corpus.stopwords.words('english')
    with open ("/Users/markusio/Library/Mobile Documents/com~apple~CloudDocs/Studies/AdvancedDataAnalysis/Project/YoutubeDataAnalysis/DataAnalysisYoutube/dataanalysis/cleaning/stopwords.txt", "r") as f:
        data = f.readlines()
        for i in data:
            stopwords_list.append(i)
    for sentence_tokens in tokenized_comments:
        filtered_words = [token for token in sentence_tokens if token not in stopwords_list]
        filtered_comments.append(filtered_words)

    #Remove all empty arrays as a result of the stopwords removal
    filtered_comments = [sentence for sentence in filtered_comments if sentence != []]
    return filtered_comments