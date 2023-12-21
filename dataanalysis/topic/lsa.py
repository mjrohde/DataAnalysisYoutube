from sklearn.decomposition import TruncatedSVD
from vectorization.tf_idf import VectorizationTfIdf

''' Performs Latent Semantic Analysis

    Takes a list of pre-processed comments, and creates a document-term
    matrix before performing the Singular Value Decomposition and 
    and reduces the dimensionality to 5 dimensions, the algorithm iterates
    100 times to handle sparce matrices. After, it extracts
    the most prevalent topics for each of the comments.
'''

def lsa(comments):
    tf_idf = VectorizationTfIdf(comments)
    lsa_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100)
    lsa = lsa_model.fit_transform(tf_idf)
    for i,topic in enumerate(lsa):
        print(f"Topic {i}: {topic*100}")