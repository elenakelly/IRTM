#pandas, numpy, matplotlib, scipy, sklearn, matplotlib, nltk

from __future__ import division
import codecs
import re
import copy
import collections

import numpy as np
#import pandas as pd
#import sklearn
#import tensorflow as tf

import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WordPunctTokenizer

#import matplotlib.pyplot as plt



nltk.download('stopwords')
nltk.download('all')
from nltk.corpus import stopwords

#Read the data

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




