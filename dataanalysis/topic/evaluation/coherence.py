from sklearn.decomposition import LatentDirichletAllocation
from gensim.models.coherencemodel import CoherenceModel

def ComputeCoherence(dictionary, texts, vectorization, limit, start=2, step=1):
    ''' Computes Coherence score

    Iterates through a range and applies the number to the number of topics for an lda model. Extracts the topics
    and sorts the list of the 100 most common words in each topic. It then computes the coherence score, using the U MASS
    model. 

    Parameters
    ----------
    dictionary : The vocabulary of the corpus

    texts: A list of the documents, pre-processed

    vectorization : An n*n matrix gathered as a result of TF-ID or BoW

    limit : An integer describing the maximum number of topics that should be used

    start : An integer describing the least number of topics to use for calculation
    
    step : The number of hops the loop should do, i.e., if start = 2, the next iteration wil have 8 topics with step=6

    Returns
    --------

    coherence_values : A list of the computed coherence values
    '''
    coherence_values = []
    top_words = []
    number_words = 100
    current_iteration = 0
    number_of_iterations = (limit - start) // step + (1 if (limit - start) % step != 0 else 0)
    print('Starting Coherence Analysis...')
    for num_topics in range(start, limit, step):
        lda_model = LatentDirichletAllocation(n_components=num_topics, learning_method='online', random_state=0, max_iter=50)
        lda_model.fit_transform(vectorization)
        topics = lda_model.components_
        feature_names = [dictionary[i] for i in range(len(dictionary))]
        #Sorts the feature names gotten from the vectorization phase, and sorts it to get the 100 most common words.
        for topic in topics:
            top_words.append([feature_names[i] for i in topic.argsort()[:-number_words - 1:-1]])
        cm = CoherenceModel(topics=top_words, texts=texts, dictionary=dictionary, coherence='u_mass')
        coherence_values.append(cm.get_coherence())
        current_iteration += 1
        print(f'{(current_iteration/number_of_iterations)*100}‰')
    return coherence_values
