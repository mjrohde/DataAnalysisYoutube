import pandas as pd
from utils.converter import convertArrayString
from sklearn.feature_extraction.text import CountVectorizer

'''
    Takes a pre-processed list of comments and performs a Bag-of-Words vectorization
    using CountVectorizer from scikit-learn
'''
def VectorizationBOW(processed_comments):
    string_array = convertArrayString(processed_comments)
    vect = CountVectorizer(ngram_range=(2,2))
    matrix = vect.fit_transform(string_array)
    df_output = pd.DataFrame(data = matrix.toarray(), columns = vect.get_feature_names_out())
    return df_output.T.tail(20)