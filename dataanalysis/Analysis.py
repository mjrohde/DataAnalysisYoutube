import pandas as pd
from gensim.models import Word2Vec

from cleaning.tokenize import tokenize_comments
from cleaning.stopwords import remove_stopwords
from cleaning.stemming import stemming

from topic.lsa import lsa
from topic.lda import lda

def load_csv(filepath):

    '''Loads the data

    Uses Pandas to load the csv into memory and extracts the
    actual comments column from the loaded data. It then produces
    a DataFrame for easier visualization of the data.

    Parameters
    ----------
    filepath : a string of the path to the csv file

    Returns
    -------
    df.head() : A dataframe, specifying the number of rows to be 
    expected
    '''
    data = pd.read_csv(filepath, usecols=['Comment (Actual)'])
    df = pd.DataFrame(data=data)
    return df.head(len(data))



def clean_text():

    ''' Cleans the text

    Performs both removal of punctuations and converts the document
    to lowercase. Performs tokenization to produce n-grams, in this case, 
    bigrams. Removes the stopwords from the corpus for each document. Stems
    the words to remove the suffixes for better and more accurate topic
    modelling.

    
    '''
    youtube_comments = []
    dataset = load_csv("../Dataset/YT_Videos_Comments.csv")
    youtube_comments = dataset['Comment (Actual)']

    #Tokenize the comments
    tokenized_comments = tokenize_comments(youtube_comments)

    #Remove stop words from the comments
    stopped_comments = remove_stopwords(tokenized_comments)

    #Stemming
    stemmed_comments = stemming(stopped_comments)

    
    #vectorized_bow = VectorizationTfIdf(stemmed_comments)  
    lsa_result = lsa(stemmed_comments, 2)
    #lda_result = lda(stemmed_comments, 1)

    return lsa_result
    
            
        

if __name__ == '__main__':
    clean_text()

