import matplotlib as plt
from wordcloud import WordCloud
from utils.converter import convertArrayString

def wordcloud_image(comments):
    string_array = convertArrayString(comments)
    long_string = ','.join(i for i in string_array)

    print(long_string)

    wordcloud = WordCloud(background_color="white", max_words=5000, contour_color=5)
    wordcloud.generate(long_string)
    plt.imshow(wordcloud)
    plt.show()