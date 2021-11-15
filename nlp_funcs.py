import numpy as np
import re
import os
import string
import nltk
import codecs
from colour import Color
from nltk.stem.porter import *
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import xml.etree.cElementTree as ET
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

# system os function
def filelist(root: str):
    """
    :param root: the pathway of data dir
    :return: Return a fully-qualified list of filenames under root directory
    """
    allfiles = []
    for path, subdirs, files in os.walk(root):
        for name in files:
            if name.endswith('txt'):
                allfiles.append(os.path.join(path, name))
    return allfiles


def get_text(filename):
    """
    Load and return the text of a text file, assuming latin-1 encoding
    :param filename: fully-qualified filename
    :return: raw text
    """
    f = codecs.open(filename, encoding='latin-1', mode='r')
    text = f.read()
    f.close()
    return text


# Preprocessing for the raw texts
def text_convert(text: str, xml=False):
    """
    Can add different format text, such as json / csv / tsv
    :param text: raw text content
    :param xml: if xml format, Parse xmltext and return the text from <tag1> and <tag2> tags
    :return: extract text
    """
    if xml:
        xmltext = text.encode('ascii', 'ignore')
        contents = []
        root = ET.fromstring(xmltext)
        contents.append(root.find('tag1').text)
        for child in root.iterfind('.//tag2/*'):
            contents.append(child.text)
        text = ' '.join(contents)
    return text


def load_corpus(contents: list, labels: list):
    """
    Create a dictionary of (key, text) associations.
    :param contents: a list of text
    :param labels: a list of key / ID
    :return: a dict of pairs
    """
    corpus = {}
    for text, key in zip(contents, labels):
        corpus[key] = text
    return corpus


def tokenize(text: str, n: int):
    """
    Tokenize text and return a non-unique list of tokenized words
    found in the text. Normalize to lowercase, strip punctuation,
    remove stop words, drop words of length < 3, strip digits.
    :param text: raw text
    :param n: need to longer than n word
    :return: a list of token word filter by stopwords
    """
    text = text.lower()
    text = re.sub('[' + string.punctuation + '0-9\\r\\t\\n]', ' ', text)
    # nltk.download()
    tokens = nltk.word_tokenize(text)
    good_words = [w for w in tokens if len(w) > n if w not in stopwords()]
    return good_words


def stopwords():
    """
    Can design stop words set here.
    default stop word from sklearn
    :return: a list of stopwords
    """
    return ENGLISH_STOP_WORDS


def stemwords(words: list):
    """
    :param words: Given a list of tokens/words
    :return: a new list with each word stemmed using a PorterStemmer
    """
    stemmer = PorterStemmer()
    stem_words = [stemmer.stem(w) for w in words]
    return stem_words


def tokenizer(text: str):
    """
    :param text: raw text
    :return: a new list with each stemmed / filtered / lower words
    """
    return stemwords(tokenize(text, 0))


# GLoVe word embedding
def load_glove(pwd: str):
    """
    :param pwd: pathway argument of data dir
    :return: dict with word/array pair
    """
    dict_map = {}
    dict_sub = {}
    pwd_vocab = str(pwd) + "/glove_vocab.txt"
    pwd_vector = str(pwd) + "/glove_vector.npy"
    vecs = np.load(pwd_vector)
    index = 0
    f_vocab = open(pwd_vocab, "r", encoding='utf-8')
    for line in f_vocab.readlines():
        dict_map[line.strip('\n')] = vecs[index]
        index += 1
    f_vocab.close()
    return dict_map
    # limit the word in glove
    # pwd = '/usr/share/dict/words'
    # f = open(pwd, "r", encoding='utf-8')
    # r_word = list(line.strip('\n').lower() for line in f.readlines())
    # f.close()
    # for word in r_word:
    #     try:
    #         dict_sub[word] = dict_map[word]
    #     except BaseException:
    #         pass
    # return dict_sub


def closest_words(gloves: dict, word: str, n: int):
    """
    :param gloves: dict of word: word vector pairs
    :param word: search word term
    :param n: the number of return closest word
    :return: list of words
    """
    l1 = []
    array_target = gloves[word]
    for key, value in gloves.items():
        if word != key:
            distance = wv_dis(array_target, value)
            t = (distance, key)
            l1.append(t)
    l1.sort()
    return list(l1[i][1] for i in range(n))


def analogies(gloves: dict, x: str, y: str, z: str, n: int):
    """
    :param gloves: dict of word: word vector pairs
    :param x, y, z: use the relationship between x and y find the potential same relation for z
    :param n: the number of return closest word
    :return: list of words
    """
    l1 = []
    x_target = gloves[x]
    y_target = gloves[y]
    z_target = gloves[z]
    xy = x_target - y_target
    for key, value in gloves.items():
        if z != key:
            zv = z_target - value
            vec_diff = vector_dis(xy, zv)
            t = (vec_diff, key)
            l1.append(t)
    l1.sort()
    return list(l1[i][1] for i in range(n))


def wv_dis(v1: np.array, v2: np.array):
    """
    :param v1, v2: word vector
    :return: the distance of vector
    """
    return np.linalg.norm(v1 - v2)


def wv_centroid(words: list, gloves: dict):
    """
    Return the word vector centroid for the text. Sum the word vectors
    for each word and then divide by the number of words. Ignore words
    not in gloves.
    :param words: a list of word after filter
    :param glove: a dict of word vector
    :return: the word vector centroid for the text
    """
    v_sum = np.zeros(shape=(len(gloves['the'])))
    count = 0
    for word in words:
        try:
            v_sum += gloves[word]
            count += 1
        except BaseException:
            pass
    return v_sum / count


def wv_centroid_recommend(article: tuple, articles: list, n: int):
    """
    :param article: text ID and word vector centroid pair
    :param articles: a list of all text ID and word vector centroid pairs
    :return: return a list of the n recommend closest to article's distance and ID
    """
    res = []
    wv_target = article[1]
    for a in articles:
        wv_a = a[1]
        distance = wv_dis(wv_target, wv_a)
        res.append((distance, a))
    res.sort(key=lambda x: x[0])
    res = res[1:n+1]
    return list((elem[0], elem[1][0]) for elem in res)


# TFIDF
def compute_tfidf(corpus: dict):
    """
    Create and return a TfidfVectorizer object before training it.
    Call fit() on the list of document strings, which figures out
    all the inverse document frequencies (IDF) for use later by
    the transform() function.
    :param corpus: a dict of key and text pairs
    :return: a untrained tfidf object model
    """
    tfidf = TfidfVectorizer(input='content',
                            analyzer='word',
                            preprocessor=gettext,
                            tokenizer=tokenizer,
                            stop_words='english',
                            decode_error='ignore')
    contents = corpus.values()
    tfidf.fit(contents)
    return tfidf


def summarize_tfidf(tfidf, text: str, n: int):
    """
    Given a trained TfidfVectorizer object and some XML text, return
    up to n (word,score) pairs in a list. Discard any terms with
    scores < 0.09.
    :param tfidf: a trained tfidf model
    :param text: text from corpus to transform
    :param n: the number of (word,score) pairs in a list
    :return: return up to n (word,score) pairs in a list
    """
    res = []
    tfidf_matrix = tfidf.transform([text])
    word_index = tfidf_matrix.nonzero()[1]
    word_list = tfidf.get_feature_names()
    tfidf_wands = [(word_list[i], tfidf_matrix[0, i]) for i in word_index]
    for word, score in tfidf_wands:
        if score >= 0.09:
            res.append((word, score))
    res.sort(reverse=True, key=lambda x: x[1])
    res = res[:n]
    return res


# Sentiment Intensity Analyzer
def neg_pos_score(articles: list):
    """
    :param text: a list of dict including article's feature (title, text)
    :return: add sentiment intensity score in range [-1, 1]
    """
    SIA = SentimentIntensityAnalyzer()
    for article in articles:
        article['score'] = SIA.polarity_scores(text=article['text'])['compound']
    return


def add_color(articles):
    """
    Given a list of articles, add a "color" key to each dictionary with a value
    containing a color graded from red to green. Pure red would be for -1.0
    sentiment score and pure green would be for sentiment score 1.0.
    :param articles: a list of dict including article's feature (title, text, score)
    :return: the median of all sentiment score
    """
    colors = list(Color("red").range_to(Color("green"), 100))
    n = len(colors) - 1
    senti_score = []
    for article in articles:
        score = article['score']
        senti_score.append(score)
        c_idx = int((score + 1) * n / 2)
        article['color'] = colors[c_idx]
    return np.median(senti_score)
