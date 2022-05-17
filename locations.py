
import re
from nltk.corpus import stopwords
import networkx as nx
import en_core_web_lg
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import codecs 


all_locations = pd.DataFrame(columns=['locations','freq','novel'])
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

    location = [ent.text for ent in doc.ents if ent.label_ == 'LOC']

    print("Finding the locations and the locations...")
    #print("date  ",date )
    #print("location ",location)
    print("\n")


    index_to_delete=[]
    list_of_locations=location[0:len(location)]

    for i in range(len(list_of_locations)):
        list_of_locations[i]=re.sub(r'\'s','',list_of_locations[i])
        list_of_locations[i].strip()
        list_of_locations[i]=re.sub(r'Mrs|Mr|Ms','',list_of_locations[i]).strip()
        if re.findall(r'[^a-zA-Z_ ]',list_of_locations[i])!=[]:
            index_to_delete.append(i)
        elif len(re.split('\W',list_of_locations[i]))<2:
            index_to_delete.append(i)     
    for index in sorted(index_to_delete, reverse=True):
        del list_of_locations[index]
    unique_char=list(set(list_of_locations))
    
    #print("list of persons ",unique_char)
    print("\n")

    freq = []
    for i in unique_char:
        freq.append(list_of_locations.count(i))
    characters = pd.DataFrame(unique_char, columns=['locations'])
    characters['freq']=freq
    characters['novel']= re.split(r'\.',re.split(r'Stephen_King_',path)[1])[0]
    characters.sort_values(by='freq', ascending=False, inplace=True)
    print("characters ",characters)
    print("\n")

    all_locations = all_locations.append(characters,ignore_index=True)
    print(all_locations)
    #all_characters = all_characters.pd.concat(characters,ignore_index=True)

print(all_locations)
print("END")

#Collect characters from all novels
print("Collect characters from all novels")
print("\n")
all_locations.sort_values(by='novel', ascending=False, inplace=True)
print(all_locations.head())
all_locations.to_csv('csv/all_locations.csv', index=False)

### Connection between books

mentions = pd.read_csv('csv/all_locations.csv')
mentioned_locations = list(mentions.locations)

print(mentioned_locations)
locations = all_locations[all_locations['locations'].isin(mentioned_locations)].sort_values(by = 'locations')
print(locations)


locations_connections = pd.DataFrame(columns=['novel','neighbour','strength'])
k = sorted(list(set(all_locations.novel)))
print("char", k)
print("\n")


for i in range (len(k)):
    for j in range (i+1, len(k)):
        c = locations[(locations['novel']==k[i])| (locations['novel'] == k[j])].groupby(by='locations').count()
        s = len(c[(c['novel']>1)])
        locations_connections.loc[len(locations_connections)] = [k[i],k[j],s]

print("Character connetion",locations_connections)
locations_connections.to_csv('csv/locations_connections.csv', index=False)

G = nx.Graph()
for index, row in locations_connections.iterrows():
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

plt.savefig('pngs/locations.png')
l,r = plt.xlim()
plt.xlim(l-0.2,r+0.2)
plt.savefig('pngs/locations1.png')