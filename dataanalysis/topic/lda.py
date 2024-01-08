from sklearn.decomposition import LatentDirichletAllocation

from vectorization.tf_idf import VectorizationTfIdf
from vectorization.bow import VectorizationBOW

from utils.wc import generate_and_display_wordcloud



def lda(comments, index):
    '''Semantic analysis using Latent Dirichlet Allocation

    The function performs LDA with both TF-IDF and BoW. It then prints the
    connection between a specified document and the topics. The higher the score
    the more prevalent the topic is in the document.
    
    For better visualization, a wordcloud is generated, displaying the 50 most prevalent
    words in the topic.
    
    Parameters
    ----------
    Comments : List of preprocessed comments

    index : Integer for use of displaying single document and its connection
            with the topics
    '''
    tf_idf, terms = VectorizationTfIdf(comments)
    bow = VectorizationBOW(comments)
    lda_model = LatentDirichletAllocation(n_components=10, learning_method='online', random_state=42, max_iter=1)
    document_topic_probabilities = lda_model.fit_transform(tf_idf)

    #Prints the coherence between the topics and the specified document
    print(f"Document: {index}")
    for topic_index, topic_probability in enumerate(document_topic_probabilities[index]):
        print(f"Topic {topic_index}: {topic_probability * 100}")
    print("\n")
    
    #Prints the topics and the 10 most prevalent words in descending order.
    #Inspiration: https://www.kaggle.com/code/rajmehra03/topic-modelling-using-lda-and-lsa-in-sklearn/notebook
    for topic_index, topic_vector in enumerate(lda_model.components_):
        word_vector = zip(terms, topic_vector)
        top_words = sorted(word_vector, key=lambda x: x[1], reverse=True)[:10]
        topic_words_str = ' '.join(word[0] for word in top_words)
        print(f"Topic {topic_index}: {topic_words_str}\n")
    
    # Creates a wordcloud with the 30 first words existing in the topic at the specified index
    generate_and_display_wordcloud(lda_model.components_[index], terms)
