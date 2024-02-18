def convertArrayString(stemmed_array):
    
    ''' Converts a list of lists to a list of strings

    Adds all bigrams within the same document into a string 
    for use in TF-IDF, BoW, and semantic analysis

    Parameters
    ----------
    stemmed_array : A list of pre-processed comments

    Returns
    ------- 
    processed_strings : A list of comments converted from bigram to a single string
                        for vectorization
    '''
    processed_strings = []
    comment = ''
    for stemmed_word_array in stemmed_array:
        comment = ''
        for word in stemmed_word_array:
            comment += word + " "
        processed_strings.append(comment.strip())
    return processed_strings