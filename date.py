
import re
from nltk.corpus import stopwords
import networkx as nx
import en_core_web_lg
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import codecs 


all_dates = pd.DataFrame(columns=['dates','freq','novel'])
text_paths = glob.glob("data/*.txt")
books = []
print(text_paths)
for path in text_paths:
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        books.append(text)
    print(path)
    print("Reading books...")
    print("\n")
    for novel in books:
        with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
            book = f.read()
    f.close()
    nlp = en_core_web_lg.load()
    nlp.max_length = 100000000
    doc = nlp(book)
    print("Reading.... ", path)
    print("\n")

    date = [ent.text for ent in doc.ents if ent.label_ == 'DATE']

    print("Finding the locations and the dates...")
    #print("date  ",date )
    #print("location ",location)
    print("\n")


    index_to_delete=[]
    list_of_dates=date[0:len(date)]

    for i in range(len(list_of_dates)):
        list_of_dates[i]=re.sub(r'\'s','',list_of_dates[i])
        list_of_dates[i].strip()
        if re.findall(r'[^a-zA-Z_ ]',list_of_dates[i])!=[]:
            index_to_delete.append(i)
        elif len(re.split('\W',list_of_dates[i]))<2:
            index_to_delete.append(i)     
    for index in sorted(index_to_delete, reverse=True):
        del list_of_dates[index]
    unique_char=list(set(list_of_dates))
    
    #print("list of persons ",unique_char)
    print("\n")

    freq = []
    for i in unique_char:
        freq.append(list_of_dates.count(i))
    characters = pd.DataFrame(unique_char, columns=['dates'])
    characters['freq']=freq
    characters['novel']= re.split(r'\.',re.split(r'Stephen_King_',path)[1])[0]
    characters.sort_values(by='freq', ascending=False, inplace=True)
    print("characters ",characters)
    print("\n")

    all_dates = all_dates.append(characters,ignore_index=True)
    print(all_dates)
    #all_characters = all_characters.pd.concat(characters,ignore_index=True)

print(all_dates)
print("END")
print( "characters ",characters)
#Collect characters from all novels
print("Collect characters from all novels")
print("\n")
all_dates.sort_values(by='freq', ascending=False, inplace=True)
print(all_dates.head())
all_dates.to_csv('csv/all_dates.csv', index=False)



### Connection between books

mentions = pd.read_csv('csv/all_dates.csv')
print(mentions.head())

mentioned_dates = list(mentions.dates)
#(list(set(mentions.dates)))
print(mentioned_dates)
dates = all_dates[all_dates['dates'].isin(mentioned_dates)].sort_values(by = 'dates')
print(dates)

dates_connections = pd.DataFrame(columns=['novel','neighbour','strength'])
k = sorted(list(set(all_dates.novel)))
print("char", k)
print("\n")

for i in range (len(k)):
    for j in range (i+1, len(k)):
        c = dates[(dates['novel']==k[i])| (dates['novel'] == k[j])].groupby(by='dates').count()
        s = len(c[(c['novel']>1)])
        dates_connections.loc[len(dates_connections)] = [k[i],k[j],s]

print("dates connetion",dates_connections)
dates_connections.to_csv('csv/dates_connections.csv', index=False)


G = nx.Graph()
for index, row in dates_connections.iterrows():
    if row['strength']>0:
        G.add_edge(row['novel'],row['neighbour'],weight=row['strength'])
        
weights = nx.get_edge_attributes(G,'weight').values()
weights = [float(i)/max(list(weights)) for i in list(weights)]
pos = nx.circular_layout(G)


nx.draw(G,
        pos,
        with_labels=True, 
        width=list(weights))

import matplotlib.pyplot as plt


l,r = plt.xlim()
plt.xlim(l-0.2,r+0.2)
plt.savefig('dates.png')
