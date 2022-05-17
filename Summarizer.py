import nltk
from nltk.corpus import stopwords
#nltk.download('punkt')
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import re
import gensim
from nltk.tokenize import word_tokenize, sent_tokenize


##############################
def read_file(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
        sentences = sample_sentence_splitter(text)
    return sentences

# Dummy splitter - split sentence by detecting words with 1st letter upper-case
def sample_sentence_splitter(text):
    return re.split(" ([A-Z][^A-Z]*)", text)


def sentence_similarity(sentence1, sentence2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sentence1 = [w.lower() for w in sentence1]
    sentence2 = [w.lower() for w in sentence2]
    all_words = list(set(sentence1+sentence2))
    vector1 = [0]*len(all_words)
    vector2 = [0]*len(all_words)

    for w in sentence1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sentence2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    # print("cosine distance=", cosine_distance(vector1, vector2))
    return cosine_distance(vector1, vector2)


def generate_similarity_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for index1 in range(len(sentences)):
        for index2 in range(len(sentences)):
            if index1==index2:
                continue
            similarity_matrix[index1][index2] = sentence_similarity(sentences[index1], sentences[index2], stop_words)
    return similarity_matrix


def generate_summary(file, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    sentences = read_file(file)
    sentence_similarity_matrix = generate_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentences[i][1]))
    print("Summary \n", ". ".join(summarize_text))

#########################################

# Split sentences into words
def sentences_to_words(file):
    with open(file, 'r') as f:
        text = f.read()
        tokens = word_tokenize(text)
    return tokens

# Tokenize sentences
def tokenize_setences(file):
    file_docs = []
    with open(file, 'r') as f:
        tokens = sent_tokenize(f.read())
        for line in tokens:
            file_docs.append(line)
    return file_docs

# Tokenize words and create a dictionary
def create_dictionary(path):
    file_docs = read_file(path)
    gen_docs = [[w.lower() for w in word_tokenize(text)]
                for text in file_docs]
    dictionary = gensim.corpora.Dictionary(gen_docs)
    return dictionary

s = read_file('./data/Stephen_King_Carrie.txt')
generate_summary('./data/Stephen_King_Carrie.txt')
# #words = sentences_to_words('./data/Stephen_King_Carrie.txt')
# #sss = tokenize_setences('./data/Stephen_King_Carrie.txt')
# dict = create_dictionary('./data/Stephen_King_Carrie.txt')
# for index, el in dict:
#     print(el)

