
''' Converts a list of lists to a list of strings

    Adds all bigrams within the same document into a string 
    for use in TF-IDF, BoW, and semantic analysis
'''

def convertArrayString(stemmed_array):
    string_array = []
    comment = ''
    for array in stemmed_array:
        for word in array:
            temp_comment = ' '.join(i for i in word if i not in comment)
            if temp_comment == '': 
                comment += ""
            else:
                comment += temp_comment + " "
        string_array.append(comment.strip())
        comment = ""
    return string_array