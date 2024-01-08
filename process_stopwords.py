with open("stopwords.txt", "r") as f:
    data = f.readlines()

stopword_list = []
for i in data:
    string = " ".join(i.split())
    stopword = string.split(" ")[1]
    print(stopword)
    stopword_list.append(stopword)

with open("stopwords.txt", "w+") as f:
    for i in stopword_list:
        f.write(f"{i}\n")
    f.close()

    