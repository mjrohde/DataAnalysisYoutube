import matplotlib.pyplot as plt
from wordcloud import WordCloud


def generate_and_display_wordcloud(topic_vector, feature_names):
    ''' Creates a wordcloud 
    
    Creates a wordcloud based on a specified topic which is obtained
    through semantic analysis techniques.

    Parameters
    ----------
    topic_vector : The vector representation of a topic obtained through LSA or LDA

    feature_names : A list of the feature names extracted from TF-IDF vectorization
    '''
    topic_words = ' '.join(word[0] for word in sorted(zip(feature_names, topic_vector), key=lambda x: x[1], reverse=True)[:30])
    wordcloud = WordCloud()
    wordcloud.generate(topic_words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()