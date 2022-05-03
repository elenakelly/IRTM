
#libraries needed pandas, numpy, matplotlib, scipy, sklearn, matplotlib, nltk

from __future__ import division
import codecs
import re
import copy
import collections

import numpy as np
import pandas as pd

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

import matplotlib.pyplot as plt




nltk.download('stopwords')
nltk.download('all')
from nltk.corpus import stopwords

### READ DATA ###

with codecs.open('data/Stephen_King_BagOfBones.txt', 'r', encoding='utf-8', errors='ignore') as f:
    bag_of_bones = f.read()
with codecs.open('data/Stephen_King_BlackHouse.txt', 'r', encoding='utf-8', errors='ignore') as f:
    black_house = f.read()
with codecs.open('data/Stephen_King_Carrie.txt', 'r', encoding='utf-8', errors='ignore') as f:
    carrie = f.read()
with codecs.open('data/Stephen_King_It.txt', 'r', encoding='utf-8', errors='ignore') as f:
    it = f.read()
with codecs.open('data/Stephen_King_Misery.txt', 'r', encoding='utf-8', errors='ignore') as f:
    misery = f.read()
with codecs.open('data/Stephen_King_SalemsLot.txt', 'r', encoding='utf-8', errors='ignore') as f:
    salems_lot = f.read()
with codecs.open('data/Stephen_King_SongOfSusannah.txt', 'r', encoding='utf-8', errors='ignore') as f:
    song_of_susannah = f.read() 
with codecs.open('data/Stephen_King_TheDarkTower.txt', 'r', encoding='utf-8', errors='ignore') as f:
    the_dark_tower = f.read()
with codecs.open('data/Stephen_King_TheDeadZone.txt', 'r', encoding='utf-8', errors='ignore') as f:
    the_dead_zone = f.read()
with codecs.open('data/Stephen_King_TheDrawingOfTheThree.txt', 'r', encoding='utf-8', errors='ignore') as f:
    the_drawing_of_the_three = f.read()
with codecs.open('data/Stephen_King_TheGirlWhoLovedTomGordon.txt', 'r', encoding='utf-8', errors='ignore') as f:
    the_girl_who_loved_tom_gordon = f.read()
with codecs.open('data/Stephen_King_TheGunslinger.txt', 'r', encoding='utf-8', errors='ignore') as f:
    the_gunslinger = f.read()
with codecs.open('data/Stephen_King_TheLittleSistersOfEluria.txt', 'r', encoding='utf-8', errors='ignore') as f:
    the_little_sisters_of_eluria = f.read()
with codecs.open('data/Stephen_King_TheShining.txt', 'r', encoding='utf-8', errors='ignore') as f:
    the_shining = f.read()
with codecs.open('data/Stephen_King_TheWasteLands.txt', 'r', encoding='utf-8', errors='ignore') as f:
    the_waste_lands = f.read()
with codecs.open('data/Stephen_King_WizardAndGlass.txt', 'r', encoding='utf-8', errors='ignore') as f:
    wizard_and_glass = f.read()

### PROCESS DATA ###

#english stop words
esw = stopwords.words('english')
esw.append('would')

#filter tokens with regular expressions
word_pattern = re.compile("^\w+$")

#token counter function
def token_counter(text):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(PorterStemmer().stem(text))
    tokens = list(map(lambda x: x.lower(), tokens))
    tokens = [token for token in tokens if re.match(word_pattern, token) and token not in esw]
    '''tokens = [token for token in tokens if word_pattern.match(token)]
    tokens = [token for token in tokens if token not in esw]
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token not in ['.', ',', '!', '?']]
    tokens = [token for token in tokens if len(token) > 1]'''
    return collections.Counter(tokens), len(tokens)

#absolute frequency and relative frequency of most common words
def word_frequency(counter,text):
    abs_freq = np.array([el[1] for el in counter])
    rel_freq = abs_freq/text
    index = [el[0] for el in counter]
    df = pd.DataFrame(data=np.array([abs_freq,rel_freq]).T, index=index, columns=['Absolute Frequency','Relative Frequency'])
    df.index.name = "Most common words"
    return df

### ANALYSIS ### 

#calculate & save most common words for each book 
bag_of_bones_counter, bag_of_bones_size = token_counter(bag_of_bones)
bag_of_bones_df = word_frequency(bag_of_bones_counter.most_common(100), bag_of_bones_size)
bag_of_bones_df.to_csv('data/bag_of_bones_df.csv')

#for all the books, calculate the most common words
black_house_counter, black_house_size = token_counter(black_house)
black_house_df = word_frequency(black_house_counter.most_common(100), black_house_size)
black_house_df.to_csv('data/black_house_df.csv')

carrie_counter, carrie_size = token_counter(carrie)
carrie_df = word_frequency(carrie_counter.most_common(100), carrie_size)
carrie_df.to_csv('data/carrie_df.csv')

it_counter, it_size = token_counter(it)
it_df = word_frequency(it_counter.most_common(100), it_size)
it_df.to_csv('data/it_df.csv')

misery_counter, misery_size = token_counter(misery)
misery_df = word_frequency(misery_counter.most_common(100), misery_size)
misery_df.to_csv('data/misery_df.csv')

salems_lot_counter, salems_lot_size = token_counter(salems_lot)
salems_lot_df = word_frequency(salems_lot_counter.most_common(100), salems_lot_size)
salems_lot_df.to_csv('data/salems_lot_df.csv')

song_of_susannah_counter, song_of_susannah_size = token_counter(song_of_susannah)
song_of_susannah_df = word_frequency(song_of_susannah_counter.most_common(100), song_of_susannah_size)
song_of_susannah_df.to_csv('data/song_of_susannah_df.csv')

the_dark_tower_counter, the_dark_tower_size = token_counter(the_dark_tower)
the_dark_tower_df = word_frequency(the_dark_tower_counter.most_common(100), the_dark_tower_size)
the_dark_tower_df.to_csv('data/the_dark_tower_df.csv')

the_dead_zone_counter, the_dead_zone_size = token_counter(the_dead_zone)
the_dead_zone_df = word_frequency(the_dead_zone_counter.most_common(100), the_dead_zone_size)
the_dead_zone_df.to_csv('data/the_dead_zone_df.csv')

the_drawing_of_the_three_counter, the_drawing_of_the_three_size = token_counter(the_drawing_of_the_three)
the_drawing_of_the_three_df = word_frequency(the_drawing_of_the_three_counter.most_common(100), the_drawing_of_the_three_size)
the_drawing_of_the_three_df.to_csv('data/the_drawing_of_the_three_df.csv')

the_girl_who_loved_tom_gordon_counter, the_girl_who_loved_tom_gordon_size = token_counter(the_girl_who_loved_tom_gordon)
the_girl_who_loved_tom_gordon_df = word_frequency(the_girl_who_loved_tom_gordon_counter.most_common(100), the_girl_who_loved_tom_gordon_size)
the_girl_who_loved_tom_gordon_df.to_csv('data/the_girl_who_loved_tom_gordon_df.csv')

the_gunslinger_counter, the_gunslinger_size = token_counter(the_gunslinger)
the_gunslinger_df = word_frequency(the_gunslinger_counter.most_common(100), the_gunslinger_size)
the_gunslinger_df.to_csv('data/the_gunslinger_df.csv')

the_little_sisters_of_eluria_counter, the_little_sisters_of_eluria_size = token_counter(the_little_sisters_of_eluria)
the_little_sisters_of_eluria_df = word_frequency(the_little_sisters_of_eluria_counter.most_common(100), the_little_sisters_of_eluria_size)
the_little_sisters_of_eluria_df.to_csv('data/the_little_sisters_of_eluria_df.csv') 

the_shining_counter, the_shining_size = token_counter(the_shining)
the_shining_df = word_frequency(the_shining_counter.most_common(100), the_shining_size)
the_shining_df.to_csv('data/the_shining_df.csv')

the_waste_lands_counter, the_waste_lands_size = token_counter(the_waste_lands)
the_waste_lands_df = word_frequency(the_waste_lands_counter.most_common(100), the_waste_lands_size)
the_waste_lands_df.to_csv('data/the_waste_lands_df.csv')

wizard_and_glass_counter, wizard_and_glass_size = token_counter(wizard_and_glass)
wizard_and_glass_df = word_frequency(wizard_and_glass_counter.most_common(100), wizard_and_glass_size)
wizard_and_glass_df.to_csv('data/wizard_and_glass_df.csv')

### COMPARISON ###

all_counter = bag_of_bones_counter + black_house_counter + carrie_counter + it_counter + misery_counter +  salems_lot_counter + song_of_susannah_counter + the_dark_tower_counter + the_dead_zone_counter + the_drawing_of_the_three_counter + the_girl_who_loved_tom_gordon_counter + the_gunslinger_counter + the_little_sisters_of_eluria_counter + the_shining_counter + the_waste_lands_counter + wizard_and_glass_counter
all_df = word_frequency(all_counter.most_common(100), 1)
most_common_words = all_df.index.values

all_df.to_csv('data/all_df.csv')




