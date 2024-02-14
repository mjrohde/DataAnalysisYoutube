from sklearn.decomposition import LatentDirichletAllocation
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.lda_model as py_lda_model
import os
import sys

from vectorization.tf_idf import VectorizationTfIdf
from vectorization.bow import VectorizationBOW


from topic.evaluation.coherence import ComputeCoherence



def lda(comments, vectorization_technique):
    '''Semantic analysis using Latent Dirichlet Allocation

    The function performs Latent Dirichlet Analysis. It requires user input to determine 
    the vectorization technique to use.
    
    Parameters
    ----------
    Comments : List of preprocessed comments 

    vectorization technique: A digit (1 or 2) describing vectorization technique to use
    '''
    print('Starting Latent Dirichlet Allocation...')
    number_of_topics = 12
    lda_model = LatentDirichletAllocation(n_components=number_of_topics, learning_method='online', random_state=42, max_iter=100)
    if vectorization_technique == 'TF-IDF':
        vectorizer, dataframe, feature_names, matrix = VectorizationTfIdf(comments)
        lda_model.fit_transform(dataframe)
    else:
        vectorizer, dataframe, feature_names, matrix = VectorizationBOW(comments)
        lda_model.fit_transform(dataframe)

        
    #Prints the coherence between the topics and the specified document
    word_id_dictionary = corpora.Dictionary(comments)

    max_topics=50; start=10; step=5;
    coherence_scores = ComputeCoherence(word_id_dictionary, comments, dataframe, max_topics, start, step)
    num_topics_range = range(start, max_topics, step)
    #Plotting
    plt.plot(num_topics_range, coherence_scores)
    plt.xlabel("Num Topics")
    plt.ylabel("Coherence score")
    plt.legend(("coherence_values"), loc='best')
    plt.show()
    
    print('Finalizing...')
    panel = py_lda_model.prepare(lda_model, matrix, vectorizer, mds='tsne')
    pyLDAvis.save_html(panel, f'{sys.path[0]}/Results/LDA_{vectorization_technique}_visualization.html')

    os.open(f'{sys.path[0]}/Results/LDA_{vectorization_technique}_visualization.html')