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
    current_comment = ''
    for stemmed_word_array in stemmed_array:
        for word in stemmed_word_array:
            temp_comment = ' '.join(char for char in word if char not in current_comment)
            if temp_comment == '': 
                current_comment += ""
            else:
                current_comment += temp_comment + " "
        processed_strings.append(current_comment.strip())
        current_comment = ""
    return processed_strings