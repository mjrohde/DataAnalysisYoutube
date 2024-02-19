import sys
import webbrowser
import os

import gensim.corpora as corpora
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.lda_model as py_lda_model
from sklearn.decomposition import LatentDirichletAllocation

from vectorization.bow import VectorizationBOW
from vectorization.tf_idf import VectorizationTfIdf
from topic.evaluation.coherence import ComputeCoherence



def lda(comments, vectorization_choice, coherence_choice):
    """Semantic analysis using Latent Dirichlet Allocation

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

    coherence_choice : bool
    A boolean indcating whether the coherence values should be computed
    """
    print("Starting Latent Dirichlet Allocation...")
    number_of_topics = 20 if vectorization_choice == "BoW" else 14
    lda_model = LatentDirichletAllocation(
        n_components=number_of_topics,
        learning_method="online",
        random_state=42,
        max_iter=100,
    )
    if vectorization_choice == "TF-IDF":
        vectorizer, dataframe, document_term_matrix = VectorizationTfIdf(
            comments
        )
    else:
        vectorizer, dataframe, document_term_matrix = VectorizationBOW(
            comments
        )

    lda_model.fit_transform(document_term_matrix)

    # Prints the coherence between the topics and the specified document
    if coherence_choice:
        word_id_dictionary = corpora.Dictionary(comments)

        max_topics = 50
        start = 10
        step = 5
        coherence_scores = ComputeCoherence(
            word_id_dictionary, comments, dataframe, max_topics, start, step
        )
        num_topics_range = range(start, max_topics, step)
        # Plotting
        plt.plot(num_topics_range, coherence_scores)
        plt.xlabel("Num Topics")
        plt.ylabel("Coherence Values")
        plt.legend(("coherence_values"), loc="best")
        plt.title("Coherence Values")
        plt.show()

    print("Finalizing...")
    file_path = os.path.join(sys.path[0], 'Results', f"LDA_{vectorization_choice}_visualization.html")
    panel = py_lda_model.prepare(
        lda_model, document_term_matrix, vectorizer, mds="tsne"
    )
    pyLDAvis.save_html(
        panel, file_path
    )

    webbrowser.open_new_tab(
        f"file://{file_path}"
    )
