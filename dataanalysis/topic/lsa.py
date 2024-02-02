from sklearn.decomposition import TruncatedSVD
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora

from vectorization.tf_idf import VectorizationTfIdf
from vectorization.bow import VectorizationBOW

from utils.wc import generate_and_display_wordcloud


def lsa(comments, index):
    ''' Performs Latent Semantic Analysis

    Takes a list of pre-processed comments, and creates a document-term
    matrix before performing the Singular Value Decomposition and 
    and reduces the dimensionality to 5 dimensions, the algorithm iterates
    100 times to handle sparce matrices. After, it extracts
    the most prevalent topics for each of the comments.

    Parameters
    ----------
    comments : A preprocessed list of comments

    index : An integer used to display a single documents connection with the generated topics
    '''
    tf_idf_matrix, feature_names = VectorizationTfIdf(comments)
    bow = VectorizationBOW(comments)
    lsa_model = TruncatedSVD(n_components=3, algorithm='randomized', n_iter=100)
    lsa = lsa_model.fit_transform(tf_idf_matrix)


    #Prints the topics and the 10 most prevalent words in descending order.
    #Inspiration: https://www.kaggle.com/code/rajmehra03/topic-modelling-using-lda-and-lsa-in-sklearn/notebook
    for topic_index, topic_vector in enumerate(lsa_model.components_):
        word_vector = zip(feature_names, topic_vector)
        top_words = sorted(word_vector, key=lambda x: x[1], reverse=True)[:10]
        topic_words_str = ' '.join(word[0] for word in top_words)
        print(f"Topic {topic_index}: {topic_words_str}\n")

    #Generates a wordcloud with the 30 most prevalent words in a specified topic
    generate_and_display_wordcloud(lsa_model.components_[index], feature_names)


