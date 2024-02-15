from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import gensim.corpora as corpora
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import os
import sys

from vectorization.tf_idf import VectorizationTfIdf
from vectorization.bow import VectorizationBOW

from topic.evaluation.topic_coherence import TopicCoherence


def lsa(comments, vectorization_choice, coherence_choice):
    ''' Performs Latent Semantic Analysis

    Takes a list of pre-processed comments, and creates a document-term
    matrix before performing the Singular Value Decomposition and 
    and reduces the dimensionality to 5 dimensions, the algorithm iterates
    100 times to handle sparce matrices. After, it extracts
    the most prevalent topics for each of the comments.

    Parameters
    ----------
    comments : list
    A preprocessed list of comments

    vectorization_choice : str
    An integer used to display a single documents connection with the generated topics

    coherence_choice : bool
    A Boolean value to determine if coherence scores should be calculated
    '''
    
    
    print('Performing Latent Semantic Analysis...')
    number_of_components = 15

    lsa_model = TruncatedSVD(n_components=number_of_components, algorithm='randomized', n_iter=100)
    
    if vectorization_choice == 'TF-IDF':
        vectorizer, dataframe, feature_names, document_term_matrix = VectorizationTfIdf(comments)
    else:
        vectorizer, dataframe, feature_names, document_term_matrix = VectorizationBOW(comments)

    lsa_model.fit_transform(dataframe)
    

    #Calculates the Topic Coherence
    if coherence_choice:
        word_id_dictionary = corpora.Dictionary(comments)

        max_topics=50; start=10; step=4;
        coherence_scores = TopicCoherence(word_id_dictionary, comments, dataframe, max_topics, start, step)
        num_topics_range = range(start, max_topics, step)
        
        #Plotting the topic coherence
        plt.plot(num_topics_range, coherence_scores)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence score")
        plt.legend(("coherence_values"), loc='best')
        plt.show()

    print('Finalizing...')
    with PdfPages(f'{sys.path[0]}/Results/LSA_{vectorization_choice}_Visualization.pdf') as pdf:
        #Inspiration: https://www.kaggle.com/code/rajmehra03/topic-modelling-using-lda-and-lsa-in-sklearn/notebook
        for topic_idx, topic_weights in enumerate(lsa_model.components_):
            fig, ax = plt.subplots(figsize=(10, 8))
            word_weight_pairs = [(feature_names[i], weight) for i, weight in enumerate(topic_weights)]
            word_weight_pairs.sort(key=lambda x: x[1], reverse=True)
            top_words, top_weights = zip(*word_weight_pairs[:10])

            y_pos = np.arange(len(top_words))
            ax.barh(y_pos, top_weights, align='center')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(top_words)
            ax.invert_yaxis()  # Invert y-axis to have the highest weight on top
            ax.set_xlabel('Weight')
            ax.set_title(f'Topic {topic_idx + 1}')
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)
    os.open(f'{sys.path[0]}/Results/LSA_{vectorization_choice}_Visualization.pdf', os.O_RDONLY)
   


