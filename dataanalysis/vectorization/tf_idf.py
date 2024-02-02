from utils.converter import convertArrayString
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def VectorizationTfIdf(processed_comments):

    ''' Performs TF-IDF

    Takes a clean list of comments, and converts it to a list of strings, such that the
    TF-IDF process can be performed 

    Parameters
    ----------
    processed_comments : A list of preprocessed comments

    Returns
    -------
    tfidf_dataframe : A dataframe of the TF-IDF matrix

    tfidf_vectorizer.get_feature_names_out() : A list of all the feature names discovered during the vectorization 
    '''
    processed_strings = convertArrayString(processed_comments)
    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=100, smooth_idf=True, max_df=70, ngram_range=(2,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_strings)
    tfidf_dataframe = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
    return tfidf_dataframe, tfidf_vectorizer.get_feature_names_out()