import glob
import codecs
from nltk import pos_tag, word_tokenize
from collections import Counter
import numpy as np
import pandas as pd
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
import re
import collections
from nltk.corpus import stopwords

text_paths = glob.glob("data/*.txt")
books = []

for path in text_paths:
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        books.append(text)
    #f.close()

print("Reading books...")

def read_text(path):
    '''
    This function reads the text into python from a text file.
    '''
    with codecs.open(path, 'r', encoding='utf-8', errors='ignore') as f:
        book = f.read()
    #f.close()
    return book

#english stop words
esw = stopwords.words('english')
esw.append('would')

#filter tokens with regular expressions
word_pattern = re.compile("^\w+$")

def token_counter(text):
    tokenizer = WordPunctTokenizer()
    tokens = tokenizer.tokenize(PorterStemmer().stem(text))
    tokens = list(map(lambda x: x.lower(), tokens))
    tokens = [token for token in tokens if re.match(word_pattern, token) and token not in esw]
    tokens = [token for token in tokens if word_pattern.match(token)]
    tokens = [token for token in tokens if token not in esw]
    tokens = [token for token in tokens if not token.isdigit()]
    tokens = [token for token in tokens if token not in ['.', ',', '!', '?']]
    tokens = [token for token in tokens if len(token) > 1]
    return collections.Counter(tokens), len(tokens)

def word_frequency(counter,text):
    abs_freq = np.array([el[1] for el in counter])
    rel_freq = abs_freq/text
    index = [el[0] for el in counter]
    df = pd.DataFrame(data=np.array([abs_freq,rel_freq]).T, index=index, columns=['Absolute Frequency','Relative Frequency'])
    df.index.name = "Most common words"
    return df

def text_tokenize(book):
    '''
    This function splits words and puctuation in the block of text created in
    the 'read_text' function into one giant list where each item is a word or
    punctuation.
    '''
    tokenize = word_tokenize(book)
    #print("Tokenizing text...", tokenize)
    return tokenize

def tagging(tokenize):
    '''
    This function takes the tokenized text created
    by the text_tokenize function and tags each word with a code for the part of speech it represents
    using NLTK's algorithm.
    '''
    tagged_text = pos_tag(tokenize)
    return tagged_text

def find_proper_nouns(tagged_text):
    '''
    This function takes in the tagged text from the tagging function and Returns
    a list of words that were tagged as proper nouns. 
    '''
    proper_nouns = []
    i = 0
    while i < len(tagged_text):
        if tagged_text[i][1] == 'NNP':
                proper_nouns.append(tagged_text[i][0].lower())
        i+=1 # increment the i counter
    return proper_nouns

#english stop words
esw = stopwords.words('english')
esw.append('would')

#filter tokens with regular expressions
word_pattern = re.compile("^\w+$")


def summarize_text(proper_nouns, top_num,i):
    '''
    This function takes the proper_nouns from the list created by the
    find_proper_nouns function and counts the instances of each.  
    Using the most_common method that comes with the Counter.
    '''
    counts = dict(Counter(proper_nouns).most_common(top_num))
    #counts.to_csv('data/book_'+str(i)+'_df.csv')
    
    return counts

# This is where we call all of our functions and pass what they return to the next function

all_counter = []

for path in text_paths:
    print(path)
    a = read_text(path)
   
    counter, size = token_counter(a)
    df = word_frequency(counter.most_common(100), size)
    newpath =  path.replace(".txt", "")
    df.to_csv(str(newpath)+'_df.csv')
    all_counter += counter
    b = text_tokenize(a)
    c = tagging(b)
    d = find_proper_nouns(c)
    e = summarize_text(d, 10 ,path)

    #most_common_words = all_df.index.values()
    print(e)
    print("\n")
    
all_df = word_frequency(counter.most_common(100), 1)
all_df.to_csv('all_df.csv')
#all_df = word_frequency(counter.most_common(100), 1)
#most_common_words = all_df.index.values()

### COMPARISON ###


