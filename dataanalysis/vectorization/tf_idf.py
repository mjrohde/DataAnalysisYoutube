from utils.converter import convertArrayString
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def VectorizationTfIdf(processed_comments):

    ''' Performs TF-IDF

    Takes a clean list of comments, and converts it to a list of strings, such that the
    TF-IDF process can be performed. TF-IDF works by measuring the term frequency of a given word
    within a single document, which yields TF. The higher the TF, the less frequent the word is.

    IDF measures how rare a word is within a set of documents. it is the logarithmic quotient of the total number of documents
    and the number of documents containing a specific word

    TF-IDF is the product of TF and IDF.

    The fuunction uses TfidfVectorizer to compute the description above with the processed comments provided.
    It then creates a dataframe with the resulting numerical matrix and the corresponding feature names. 

    Parameters
    ----------
    processed_comments : A list of preprocessed comments

    Returns
    -------
    tfidf_vectorizer : A TF-IDF vectorizer object

    tfidf_dataframe : A dataframe of the TF-IDF matrix

    tfidf_vectorizer.get_feature_names_out() : A list of all the feature names discovered during the vectorization 

    tfidf_matrix : A tuple of document id and token id, with the corresponding TF-IDF score, token ids not present have a TF-IDF score of 0
    '''
    print("TF-IDF vectorization started...")
    processed_strings = convertArrayString(processed_comments)

    tfidf_vectorizer = TfidfVectorizer(use_idf=True, max_features=1000, max_df=0.7, ngram_range=(2,2))
    tfidf_matrix = tfidf_vectorizer.fit_transform(processed_strings)

    feature_names = tfidf_vectorizer.get_feature_names_out()

    tfidf_dataframe = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    print("TF-IDF vectorization done! Last step now is the Semantic analysis!")
    
    return tfidf_vectorizer, tfidf_dataframe, feature_names, tfidf_matrix