from sklearn.decomposition import TruncatedSVD
from vectorization.tf_idf import VectorizationTfIdf

def lsa(comments):
    tf_idf = VectorizationTfIdf(comments)
    lsa_model = TruncatedSVD(n_components=5, algorithm='randomized', n_iter=100)
    lsa = lsa_model.fit_transform(tf_idf)
    l = lsa[0]
    print("Review 0:")
    for i,topic in enumerate(l):
        print(f"Topic {i}: {topic*100}")