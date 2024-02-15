# Run Code

### MacOS
``` bash
cd <Your Location>/YoutubeDataAnalysis/DataAnalysisYoutube/dataanalysis
```
### Microsoft PowerShell
```powershell
cd ./<Your Location>/YoutubeDataAnalysis/DataAnalysisYoutube/dataanalysis
```
### Microsoft Command Prompt
```command prompt
cd <Your Location>/YoutubeDataAnalysis/DataAnalysisYoutube/dataanalysis
```

## Install all dependencies
``` bash
pip install -r requirements.txt
```

## Start program
```bash
python3 Analysis.py
```

# Runtime Example
The program requires user interaction to complete. It will require the user to enter the type of semantic analysis (lsa or lda) and the vectorization technique ((1)TF-IDF or (2)BoW). 
Lastly, it will ask if the program should calculate coherence scores.
The following image shows an example of input:

![Input User](Images/Working_example.png)

# Code Description

#### Coherence Scores
The coherence scores for all four models have been computed. 
As shown in the Runtime Example section above, the user can opt to perform the same calculation.
If this is unwanted, take a look at the results below.

##### LSA
LSA with BoW
![Coherence Scores BoW LSA](Images/LSA_BOW_Coherence.png)
LSA with TF-IDF
![Coherence Scores TF-IDF LSA](Images/LSA_TF-IDF_UMASS.png)

##### LDA
LDA with TF-IDF
![Coherence Scores TF-IDF LDA](Images/LDA_TFIDF_UMASS.png)
LDA with BoW
![Coherence Scores BoW LDA](Images/LDA_BOW_UMASS.png)







