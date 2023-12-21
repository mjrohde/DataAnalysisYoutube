import matplotlib as plt
from wordcloud import WordCloud
from utils.converter import convertArrayString

''' Creates a wordcloud 
    
    Creates a wordcloud based on the loaded corpus to better understand
    the documents and the corpus in general. The wordcloud is presented
    using matplotlib.
'''

def wordcloud_image(comments):
    string_array = convertArrayString(comments)
    long_string = ','.join(i for i in string_array)

    wordcloud = WordCloud(background_color="white", max_words=5000, contour_color=5)
    wordcloud.generate(long_string)

    plt.imshow(wordcloud)
    plt.show()