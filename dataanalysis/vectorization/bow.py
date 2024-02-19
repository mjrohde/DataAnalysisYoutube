import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from utils.converter import convertArrayString



def VectorizationBOW(processed_comments):

    """Performs Bag-of-Words vectorization

    Takes a pre-processed list of comments and performs a Bag-of-Words vectorization
    using CountVectorizer from scikit-learn

    Parameters
    ----------
    processed_comments : list
    A list of preprocessed comments

    Returns
    -------
    vectorizer : object
    An instance of the CountVectorizer class

    document_term_df : dataframe
    Returns a dataframe of the fitted vocabulary to the BoW model

    feature_names : nd-array
    An n-dimensional array of feature names

    term_document_matrix : array
    An array of shape (n_samples, n_features) with the result of the vectorization of the processed comments
    """
    print("BoW vectorization started...")

    # Converts the list of tokens into strings for the vectorizer
    processed_strings = convertArrayString(processed_comments)

    vectorizer = CountVectorizer(
        analyzer="word", max_features=1000, max_df=0.7, ngram_range=(2, 2)
    )
    term_document_matrix = vectorizer.fit_transform(processed_strings)

    document_term_df = pd.DataFrame(
        data=term_document_matrix.toarray(), columns=vectorizer.get_feature_names_out()
    )

    print("BoW vectorization done! Almost there now!")
    return vectorizer, document_term_df, term_document_matrix
