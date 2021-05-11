from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import threading
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import tensorflow
from tqdm import tqdm_notebook
import tensorflow_hub as hub


def get_html_soup(url):
    """Uses Beautiful Soup to extract html from a url. Returns a soup object """
    headers = {'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_6) AppleWebKit/537.36 (KHTML, like Gecko) \
                Chrome/56.0.2924.87 Safari/537.36'}
    req = Request(url, headers=headers)
    get_html_soup = BeautifulSoup(urlopen(req).read(), 'html.parser')
    return get_html_soup


class MultiThreading:

    def __init__(self, threads, iteration_list, output=[]):
        self.threads = threads
        self.output = output
        self.iteration_list = iteration_list

    def multi_thread_compile(self, thread_count, function):

        """a function that compiles an iteration list to prepare
        multi threadding"""

        jobs = []

        # distribute iteration list to batches and append to jobs list
        batches = [i.tolist() for i in np.array_split(self.iteration_list, thread_count)]

        for i in range(len(batches)):
            jobs.append(threading.Thread(target=function, args=[batches[i]]))

        return jobs

    def multi_thread_execute(self, jobs):

        """executes the multi-threading loop"""

        # Start the threads
        for j in jobs:
            j.start()

        # Ensure all of the threads have finished
        for j in jobs:
            j.join()
        return

    def Run(self, function):

        jobs = self.multi_thread_compile(self.threads, function)
        self.multi_thread_execute(jobs)


class SpacyTransformer:

    def __init__(self):
        self.df = df
        self.source_col = source_col
        self.model = model

    def fit_transform(self):
        # fit the spaCy model using multi-threading
        self.df['spaCy_doc'] = [i for i in self.model.pipe(self.df[self.source_col], n_threads=30, batch_size=1000)]
        # extract sentences
        self.df['spaCy_sentences'] = self.df['spaCy_doc'].apply(lambda x: [sent for sent in x.sents])
        # extract nouns
        self.df['spaCy_nouns'] = self.df['spaCy_doc'].apply(
            lambda x: [token.text.lower() for token in x if token.pos_ == 'NOUN'])
        # extract noun chunks
        self.df['spaCy_noun_chunk'] = self.df['spaCy_sentences'].apply(
            lambda x: [(token.root.text, token.text) for i in x for token in i.noun_chunks])
        # flatten lists into string
        pos_columns = ['spaCy_nouns']

        for i in pos_columns:
            self.df[i] = self.df[i].str.join(', ')

    def get_noun_chunks(self, noun):

        """ Function that returns a DataFrame of noun chunks for any given
        noun"""
        # temp list to append noun chunks
        temp = []
        # loop through each noun chunk
        for i in self.df['spaCy_noun_chunk']:
            # append noun and relevant noun chunk to list
            for x in i:
                temp.append((x[0], x[1]))

        # create DataFrame of data
        df = pd.DataFrame(temp)
        # column names
        df.columns = ['noun', 'noun_chunk']
        # add length of noun chunk
        df['chunk_length'] = df['noun_chunk'].map(lambda x: len(x))
        # mask on noun of interest
        df = df[df['noun'] == noun]
        # sort DF by length of noun chunk
        df = df.sort_values(by='chunk_length', ascending=False)

        # drop length column
        self.df_noun_chunk = df[['noun', 'noun_chunk']]

        return self.df_noun_chunk


def most_common_tokens(data, additional_stopwords=None, token=2):
    """This returns the most common unigrams, bigrams, trigrams that exist in the corpus"""

    # add stop words
    if additional_stopwords is None:
        additional_stopwords = []
    add_stop_words = ENGLISH_STOP_WORDS.union(additional_stopwords)
    # instantiate CountVectorizer
    vect = CountVectorizer(stop_words=add_stop_words, ngram_range=(token, token))
    # fit the data to the CountVectorizer
    X = vect.fit_transform(data)
    # get word counts
    word_counts = list(zip(vect.get_feature_names(), np.asarray(X.sum(axis=0)).ravel()))
    # create DataFrame of word counts
    word_counts = pd.DataFrame(word_counts, columns=['word', 'count'])
    # sort values
    word_counts = word_counts.sort_values(by='count', ascending=False)
    # convert to integer
    word_counts['count'] = word_counts['count'].map(lambda x: int(x))
    # only include counts where count(n) > 1
    word_counts = word_counts[word_counts['count'] > 1]
    # add a normlisation column
    word_counts['count_norm'] = word_counts['count'].map(lambda x: x / word_counts['count'].sum())
    return word_counts


class UniversalSentenceEncoder:

    def __init__(self):
        self.USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    def get_encoding(self, sentence):
        return np.array(self.USE([sentence])[0])





