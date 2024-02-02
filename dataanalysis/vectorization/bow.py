import pandas as pd
from utils.converter import convertArrayString
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def VectorizationBOW(processed_comments):

    ''' Performs Bag-of-Words vectorization

    Takes a pre-processed list of comments and performs a Bag-of-Words vectorization
    using CountVectorizer from scikit-learn

    Parameters
    ----------
    processed_comments : A list of preprocessed comments

    Returns
    -------
    document_term_df.T : The transpose of the dataframe
    '''
    processed_strings = convertArrayString(processed_comments)
    vectorizer = CountVectorizer(analyzer='word', max_features=10, ngram_range=(2,2))
    term_document_matrix = vectorizer.fit_transform(processed_strings)
    document_term_df = pd.DataFrame(data = term_document_matrix.toarray(), columns = vectorizer.get_feature_names_out())
    print(document_term_df)
    return document_term_df.T