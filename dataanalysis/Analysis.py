#!/usr/bin/env python
import pandas as pd

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

    #Allows user to choose the semantic model and the vectorization technique used for the ouptut.
    choice = True
    while choice:
        semantic_choice = input('Choose the semantic analysis: \nlsa or lda:\t')
        if semantic_choice.lower() == 'lsa' or semantic_choice.lower() == 'lda':
            choice = False
        else:
            print("Please type either lsa or lda\n\n")
    choice = True
    while choice:
        vectorization_choice = input('Choose Vectorization technique (Type 1 or 2): \nTF-IDF = 1 BoW = 2:\t').strip()
        if vectorization_choice == '1' or vectorization_choice == '2':
            if vectorization_choice == '1':
                vectorization_choice = 'TF-IDF'
            else:
                vectorization_choice = 'BoW'
            choice = False
        else:
            print('Please enter the digit 1 or 2')


    #Loading data
    youtube_comments = []
    dataset = load_csv('../Dataset/YT_Videos_Comments.csv')
    youtube_comments = dataset['Comment (Actual)']

    #Pre-processing
    tokenized_comments = tokenize_comments(youtube_comments)
    stopped_comments = remove_stopwords(tokenized_comments)
    stemmed_comments = stemming(stopped_comments)


    #Semantic Analysis
    if semantic_choice.lower() == 'lsa':
        lsa(stemmed_comments, vectorization_choice)
    elif semantic_choice.lower() == 'lda':
        lda(stemmed_comments, vectorization_choice)

    
            
        

if __name__ == '__main__':
    clean_text()

