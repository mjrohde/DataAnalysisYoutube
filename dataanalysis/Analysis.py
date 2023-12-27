import pandas as pd
import regex as re

from wordcloud import WordCloud
import matplotlib.pyplot as plt

from cleaning.tokenize import TokenizeComments
from cleaning.stopwords import RemoveStopwords
from cleaning.stemming import Stemming

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.decomposition import TruncatedSVD

from vectorization.bow import VectorizationBOW
from vectorization.tf_idf import VectorizationTfIdf

from topic.lsa import lsa
from utils.converter import convertArrayString

'''Loads the data

    Uses Pandas to load the csv into memory and extracts the
    actual comments column from the loaded data. It then produces
    a DataFrame for easier visualization of the data. 

    Returns the dataframe, specifying the number of rows to be 
    expected
'''
def load_csv(filepath):
    data = pd.read_csv(filepath, usecols=['Comment (Actual)'])
    df = pd.DataFrame(data=data)
    return df.head(100)

''' Cleans the text

    Performs both removal of punctuations and converts the document
    to lowercase. Performs tokenization to produce n-grams, in this case, 
    bigrams. Removes the stopwords from the corpus for each document. Stems
    the words to remove the suffixes for better and more accurate topic
    modelling.

'''

def clean_text():
    comments = []
    data = load_csv("/Users/markusio/Library/Mobile Documents/com~apple~CloudDocs/Studies/AdvancedDataAnalysis/Project/YoutubeDataAnalysis/Dataset/YT_Videos_Comments.csv")
    comments = data['Comment (Actual)']

    #Tokenize the comments
    tokenized_comments = TokenizeComments(comments)

    #Remove stop words from the comments
    stopped_comments = RemoveStopwords(tokenized_comments)

    #Stemming
    #stemmed_comments = Stemming(stopped_comments)

    #vectorized_bow = VectorizationTfIdf(stemmed_comments)

    #lsa_result = lsa(stemmed_comments)
    
    return stopped_comments
            
        

if __name__ == '__main__':
    clean_text()

