from sklearn.decomposition import LatentDirichletAllocation
import gensim.corpora as corpora
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.lda_model as py_lda_model
import sys
import webbrowser

from vectorization.tf_idf import VectorizationTfIdf
from vectorization.bow import VectorizationBOW


from topic.evaluation.coherence import ComputeCoherence



def lda(comments, vectorization_choice, coherence_choice):
    '''Semantic analysis using Latent Dirichlet Allocation

    The function performs Latent Dirichlet Analysis. It requires user input to determine 
    the vectorization technique to use. It uses the LDA computation provided by the scikit-learn
    library, and fits the corpus to the model provided.

    The function uses pyLDAvis to visualize the result of the computation which allows for better
    human interpretation of the results of the model. 
    
    Lastly, it opens the results in a browser, which will be the default browser of the system.

    Parameters
    ----------
    Comments : list
    A list of pre-processed comments 

    vectorization technique: str
    A digit (1 or 2) describing vectorization technique to use
    '''
    print('Starting Latent Dirichlet Allocation...')
    number_of_topics = 35
    lda_model = LatentDirichletAllocation(n_components=number_of_topics, learning_method='online', random_state=42, max_iter=100)
    if vectorization_choice == 'TF-IDF':
        vectorization_model = VectorizationTfIdf()
    else:
        vectorization_model = VectorizationBOW()

    #Apply the selected vectorization model
    vectorizer, dataframe, feature_names, document_term_matrix = vectorization_model(comments)
    lda_model.fit_transform(dataframe)

        
    #Prints the coherence between the topics and the specified document
    if coherence_choice:
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
    panel = py_lda_model.prepare(lda_model, document_term_matrix, vectorizer, mds='tsne')
    pyLDAvis.save_html(panel, f'{sys.path[0]}/Results/LDA_{vectorization_choice}_visualization.html')

    webbrowser.open_new_tab(f'file://{sys.path[0]}/Results/LDA_{vectorization_choice}_visualization.html')

