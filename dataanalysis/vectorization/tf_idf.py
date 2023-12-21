from utils.converter import convertArrayString
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

'''
    Takes a clean list of comments, and converts it to a list of strings, such that the
    TF-IDF process can be performed 
'''
def VectorizationTfIdf(processed_comments):
    string_array = convertArrayString(processed_comments)
    vect = TfidfVectorizer(use_idf=True, max_features=5, smooth_idf=True)
    model = vect.fit_transform(string_array)
    data = pd.DataFrame(model.toarray(), columns=vect.get_feature_names_out())
    return data