from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel

def ComputeCoherence(dictionary, texts, vectorization, limit, start=2, step=1):
    ''' Computes Coherence score

    Iterates through a range and applies the number to the number of topics for an lda model. Extracts the topics
    and sorts the list of the 100 most common words in each topic. It then computes the coherence score, using the U MASS
    model. 

    Parameters
    ----------
    dictionary : list
    The vocabulary of the corpus

    texts: list
    A list of the documents, pre-processed

    vectorization : array
    An n*n matrix gathered as a result of TF-ID or BoW

    limit : int
    An integer describing the maximum number of topics that should be used

    start : int
    An integer describing the least number of topics to use for calculation
    
    step : int
    The number of hops the loop should do, i.e., if start = 2, the next iteration wil have 8 topics with step=6

    Returns
    --------

    coherence_values : list
    A list of the computed coherence values
    '''
    coherence_values = []
    number_of_words = 100

    #Computes the amount of lda models that needs to be computed in the given range
    current_iteration = 0
    number_of_iterations = (limit - start + step - 1) // step
    print('Starting Coherence Analysis...')
    for num_topics in range(start, limit, step):

        #Performs LDA
        lda_model = LatentDirichletAllocation(n_components=num_topics, learning_method='online', random_state=42, max_iter=50)
        lda_model.fit_transform(vectorization)

        #Extracts the topics
        topics = lda_model.components_

        #Sorts the feature names gotten from the vectorization phase, and sorts it to get the 100 most common words.
        sorted_words = [[dictionary[i] for i in topic.argsort()[:-number_of_words - 1:-1]] for topic in topics]

        #Computes coherence scores for the LDA model with n topics
        cm = CoherenceModel(topics=sorted_words, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(cm.get_coherence())
        
        #Used to give the user an idea of how far along the process is
        current_iteration += 1
        print(f'{(current_iteration/number_of_iterations)*100}%')
    return coherence_values
