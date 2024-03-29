from gensim.models.coherencemodel import CoherenceModel
from sklearn.decomposition import TruncatedSVD


def TopicCoherence(dictionary, texts, vectorization, limit, start=2, step=1):
    """Computes Coherence score

    Iterates through a range and applies the number to the number of topics for an lsa model. Extracts the topics
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
    """
    coherence_values = []
    number_words = 100
    
    # Gives an estimate of time remaining
    current_iteration = 0
    number_of_iterations = (limit - start) // step + (
        1 if (limit - start) % step != 0 else 0
    )
    for num_topics in range(start, limit, step):

        # Generates the number of topics
        lsa_model = TruncatedSVD(
            n_components=num_topics, algorithm="randomized", n_iter=100
        )
        lsa_model.fit_transform(vectorization)

        # Extracts Topics and sorts most common words in ascending order
        topics = lsa_model.components_
        feature_names = [dictionary[i] for i in range(len(dictionary))]

        # Sorts the feature names gotten from the vectorization phase, and sorts it to get the 100 most common words.
        sorted_words = [
            [feature_names[i] for i in topic.argsort()[: -number_words - 1 : -1]]
            for topic in topics
        ]

        # Calculates coherence scores using the c_v method
        cm = CoherenceModel(
            topics=sorted_words, texts=texts, dictionary=dictionary, coherence="c_v"
        )
        coherence_values.append(cm.get_coherence())

        # Updates how far along the process is
        current_iteration += 1
        print(f"{(current_iteration/number_of_iterations)*100}%")
    return coherence_values
