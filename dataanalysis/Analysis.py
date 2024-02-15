#!/usr/bin/env python
import pandas as pd

from cleaning.tokenize import tokenize_comments
from cleaning.stopwords import remove_stopwords
from cleaning.stemming import stemming

from topic.lsa import lsa
from topic.lda import lda

def load_csv(filepath):

    ''' Loads the data

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
    return df.head(10000)

def get_user_input(prompt, choices):
    ''' Gets user input
    The user can enter their choice for the semantic model, vectorization technique, and the coherence score 
    computation. The function will run until a valid answer is given

    Parameters
    -----------
    prompt : str
    A string with the instructions for the user

    choices : dict
    a dictionary where keys are the user input, and the values are the corresponding output

    Returns
    ---------
    choices[user_input] : str
    The value from the dictionary given in the input
    '''

    while True:
        user_input = input(prompt).lower().strip()
        if user_input in choices:
            return choices[user_input]
        else:
            print('Invalid Choice. Please try again\n\n')


def get_compute_coherence_choice(prompt):
    ''' Allows user to choose if coherence score should be calculated

    The user gets the chance to answer 'y'/'n' or even 'yes'/'no' to whether the 
    coherence scores within a given range should be calculated. This is an exhaustive task, 
    and it is recommended to answer 'n'/'no' if the goal is to see the results of the current
    model.

    Parameters
    ----------
    prompt : str
    A string with the instructions for the user


    Returns
    -------
    user_choice : bool
    The value of the dictionary corresponding to the user input.
    '''

    choices = {'yes': True, 'no': False, 'y': True, 'n': False}
    user_choice = get_user_input(prompt, choices)
    return user_choice



def clean_text():

    ''' Cleans the text

    Performs both removal of punctuations and converts the document
    to lowercase. Performs tokenization to produce tokens. 
    Removes the stopwords from the corpus for each document. Stems
    the words to remove the suffixes for better and more accurate topic
    modelling.

    
    '''

    semantic_prompt = 'Choose the semantic analysis (lsa or lda):'
    semantic_choices = {'lsa': 'LSA', 'lda': 'LDA'}
    semantic_analysis = get_user_input(semantic_prompt, semantic_choices)

    vectorization_prompt = 'Choose Vectorization technique (Type 1 for TF-IDF or 2 for BoW):'
    vectorization_choices = {'1': 'TF-IDF', '2': 'BoW'}
    vectorization_technique = get_user_input(vectorization_prompt, vectorization_choices)

    coherence_prompt = 'Should the coherence score be computed? [y/n]:'
    coherence_choice = get_compute_coherence_choice(coherence_prompt)


    #Loading data
    youtube_comments = []
    dataset = load_csv('../Dataset/YT_Videos_Comments.csv')
    youtube_comments = dataset['Comment (Actual)']

    #Pre-processing
    tokenized_comments = tokenize_comments(youtube_comments)
    stopped_comments = remove_stopwords(tokenized_comments)
    stemmed_comments = stemming(stopped_comments)


    #Semantic Analysis
    if semantic_analysis.lower() == 'lsa':
        lsa(stemmed_comments, vectorization_technique, coherence_choice)
    elif semantic_analysis.lower() == 'lda':
        lda(stemmed_comments, vectorization_technique, coherence_choice)

    
            
        

if __name__ == '__main__':
    clean_text()

