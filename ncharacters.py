import re
from nltk.corpus import stopwords
import networkx as nx
import en_core_web_lg
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import glob
import codecs 


all_characters = pd.DataFrame(columns=['character','freq','novel'])
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

    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    print("Finding the characters...")
    #print("persons ",persons)
    print("\n")

    index_to_delete=[]
    list_of_p=persons[0:len(persons)]

    for i in range(len(list_of_p)):
        list_of_p[i]=re.sub(r'\'s','',list_of_p[i])
        list_of_p[i].strip()
        list_of_p[i]=re.sub(r'Mrs|Mr|Ms','',list_of_p[i]).strip()
        if re.findall(r'[^a-zA-Z_ ]',list_of_p[i])!=[]:
            index_to_delete.append(i)
        elif len(re.split('\W',list_of_p[i]))<2:
            index_to_delete.append(i)
        '''else:
            for word in re.split('\W',list_of_p[i]):
                if word in stopwords:
                    list_of_p[i]=re.sub(word,'',list_of_p[i])'''       
    for index in sorted(index_to_delete, reverse=True):
        del list_of_p[index]
    unique_char=list(set(list_of_p))
    
    #print("list of persons ",unique_char)
    print("\n")

    freq = []
    for i in unique_char:
        freq.append(list_of_p.count(i))
    characters = pd.DataFrame(unique_char, columns=['character'])
    characters['freq']=freq
    characters['novel']= re.split(r'\.',re.split(r'Stephen_King_',path)[1])[0]
    characters.sort_values(by='freq', ascending=False, inplace=True)
    print("characters ",characters)
    print("\n")

    all_characters = all_characters.append(characters,ignore_index=True)
    print(all_characters)
    #all_characters = all_characters.pd.concat(characters,ignore_index=True)

print(all_characters)
print("END")

#Collect characters from all novels
print("Collect characters from all novels")
print("\n")
#all_characters.sort_values(by='character', ascending=False, inplace=True)
print(all_characters.head())
all_characters.to_csv('csv/all_characters.csv', index=False)

#prints all the characters and their frequencies
'''k = set(list(all_characters.character))
p = list(all_characters.character)
for i in k:
    if p.count(i)<2:
        print(i, p.count(i))
        #pass'''

#example = all_characters[(all_characters['character']=='Henry Bowers')]
example = all_characters[(all_characters['character']=='Jack Sawyer')]
print("\n")
print("example ","\n",example)


#Create a network of characters

print("\n")
mentions = pd.read_csv('csv/all_characters.csv')
mentioned_character = list(mentions.character)

characters = all_characters[all_characters['character'].isin(mentioned_character)].sort_values(by = 'character')

char_connections = pd.DataFrame(columns=['novel','neighbour','strength'])
k = sorted(list(set(all_characters.novel)))
print("char", k)
print("\n")


for i in range (len(k)):
    for j in range (i+1, len(k)):
        c = characters[(characters['novel']==k[i])| (characters['novel'] == k[j])].groupby(by='character').count()
        s = len(c[(c['novel']>1)])
        char_connections.loc[len(char_connections)] = [k[i],k[j],s]

print("Character connetion",char_connections)
char_connections.to_csv('csv/char_connections.csv', index=False)

G = nx.Graph()
for index, row in char_connections.iterrows():
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
plt.savefig('pngs/charconnections.png')


# Book connections
novel_names=[]
space = " "
for i in k:
    novel_names.append(space.join(re.findall(r'[A-Z][^A-Z]*',i)))
print( "novelnames", novel_names)

novel_names.remove('It')

df=pd.DataFrame(columns=['what_novel_name','in_what_novel','mention'])

for path in text_paths:
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
    f.close()
    space=" "
    new_name = space.join(re.findall(r'[A-Z][^A-Z]*',re.split(r'\.',re.split(r'Stephen_King_',path)[1])[0]))
    print("new name ",new_name)

    for i in novel_names:
        if i != new_name:
            x=re.findall(i,text)
            if x!=[]:
                for mention in x:
                     df.loc[len(df)]=[i,new_name,mention]
    print( "paths", path)
name_connections = df.groupby(by=['what_novel_name','in_what_novel']).count()
name_connections.reset_index( inplace=True)
name_connections.columns=['novel','neighbour','strength']
print("\n")
print(name_connections)
name_connections.to_csv('csv/name_connections.csv', index=False)

G1 = nx.Graph()
for index, row in name_connections.iterrows():
    if row['strength']>0:
        G1.add_edge(row['novel'],row['neighbour'],weight=row['strength'])

edges = G1.edges()       
weights = nx.get_edge_attributes(G1,'weight').values()
weights = [float(i)/max(list(weights)) for i in list(weights)]

    
pos = nx.circular_layout(G1)
edges = G1.edges()
colors = [G1[u][v]['weight']**0.9 for u, v in edges]
cmap = matplotlib.cm.get_cmap('Blues')

nx.draw(G1,
        pos,
        with_labels=True, 
        width=list(weights),
         edge_cmap=cmap)



ax = plt.gca()
ax.margins(0.25)
plt.axis("equal")
plt.tight_layout()
plt.savefig('pngs/bookConnections.png')



    