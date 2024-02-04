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
The program requires user interaction to complete. It will require the user to enter the type of semantic analysis (lsa or lda) and the vectorization technique ((1)TF-IDF or (2)BoW). The following image shows an example of input:
![Input User](Images/Working_example.png)

# Code Description
Thorughout the code there are multiple lines that are commented out. These are highly relevant to the project. 
However, to lower the runtime, they are commented out to ensure it does not take too long to run each time. 

### Examples
#### Coherence Scores
![Coherence Scores](Images/Coherence_Scores.png)

Here, the code has been commented out. Altough this code has been run, with the following result:
![Coherence Scores Graph](Images/Coherence_Scores_Graph.png)

It is not required to run this snippet for every runtime of the program. Therefore, it has been commented out.




