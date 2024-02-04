from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel
import gensim.corpora as corpora
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import numpy as np

from vectorization.tf_idf import VectorizationTfIdf
from vectorization.bow import VectorizationBOW

from utils.wc import generate_and_display_wordcloud
from topic.coherence import ComputeCoherence



def lda(comments, index, vectorization_technique):
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
    print("Starting Latent Dirichlet Allocation...")
    lda_model = LatentDirichletAllocation(n_components=3, learning_method='online', random_state=42, max_iter=1)
    if vectorization_technique == "TF-IDF":
        tf_idf, terms = VectorizationTfIdf(comments)
        lda_model.fit_transform(tf_idf)
    else:
        bow = VectorizationBOW(comments)
        lda_model.fit_transform(bow)


    #Prints the coherence between the topics and the specified document
    '''word_id_dictionary = corpora.Dictionary(comments)
    corpus = [word_id_dictionary.doc2bow(text) for text in comments]

    max_topics=40; start=2; step=6;
    lda_models, coherence_scores = ComputeCoherence(word_id_dictionary, corpus, comments, 40, 2, 6)
    num_topics_range = range(start, max_topics, step)
    
    #Plotting
    plt.plot(num_topics_range, coherence_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()'''

    #Prints the topics and the 10 most prevalent words in descending order.
    #Inspiration: https://www.kaggle.com/code/rajmehra03/topic-modelling-using-lda-and-lsa-in-sklearn/notebook
    print("Finalizing...\n")
    for topic_index, topic_vector in enumerate(lda_model.components_):
        word_vector = zip(terms, topic_vector)
        top_words = sorted(word_vector, key=lambda x: x[1], reverse=True)[:10]
        topic_words_str = ' '.join(word[0] for word in top_words)
        print(f"Topic {topic_index + 1}: {topic_words_str}\n")
    
    # Creates a wordcloud with the 30 first words existing in the topic at the specified index
    
    generate_and_display_wordcloud(lda_model.components_, terms)
