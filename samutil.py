from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import threading
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer

from tqdm import tqdm_notebook
import tensorflow_hub as hub
import tensorflow as tf
from transformers import BertTokenizer, BertModel
import torch
from tqdm.notebook import tqdm
from sklearn.manifold import TSNE

import clip
from PIL import Image

import itertools
import networkx as nx


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

    def __init__(self, df, source_col, model):
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


class UniversalSentenceTransformer:

    def __init__(self):
        self.USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")

    def _get_encoding(self, sentence):
        return self.USE([sentence])['outputs'].numpy()

    def fit_transform(self, text_inputs=[]):
        results = []
        for text in tqdm(text_inputs):
            results.append((text, self._get_encoding(text)[0]))
        results = pd.DataFrame(results, columns=['text', 'vector'])
        vector_results = pd.DataFrame(results['vector'].apply(pd.Series))
        self.results = pd.concat([results['text'], vector_results], axis=1).set_index('text')
        self.results.columns = ['USE_' + str(i) for i in self.results.columns]
        return self.results


class BertTransformer():

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True, )
        self.model.eval()

    def _get_tokens(self, text_input):
        marked_text = "[CLS] " + text_input + " [SEP]"
        tokenized_text = self.tokenizer.tokenize(marked_text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        return indexed_tokens, segments_ids

    def _get_tensors(self, indexed_tokens, segments_ids):
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        return tokens_tensor, segments_tensors

    def _get_hidden_states(self, tokens_tensor, segments_tensors):
        with torch.no_grad():
            outputs = self.model(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]
        token_vecs = hidden_states[-2][0]
        sentence_embedding = torch.mean(token_vecs, dim=0)
        return sentence_embedding

    def fit_transform(self, text_inputs=[], reduce=True):
        results = []
        for text in tqdm(text_inputs):
            indexed_tokens, segments_ids = self._get_tokens(text)
            tokens_tensor, segments_tensors = self._get_tensors(indexed_tokens, segments_ids)
            embedding = self._get_hidden_states(tokens_tensor, segments_tensors)
            results.append((text, np.array(embedding)))
        results = pd.DataFrame(results, columns=['text', 'vector'])
        vector_results = results['vector'].apply(pd.Series)
        self.results = pd.concat([results['text'], vector_results], axis=1).set_index('text')
        self.results.columns = ['BERT_' + str(i) for i in self.results.columns]
        return self.results

class NetworkTransformer:

    def __init__(self, df):
        self.df = df
        self.edge_df = None
        self.node_df = None
        self.node_id_dic = None
        self.edge_df = None
        self.G = None
        self.graph_adjacencies = None
        self.graph_betweeness = None
        self.graph_clustering_coeff = None
        self.graph_communities = None
        self.graph_communities_dict = None

    def _get_batches(self, df):
        batch = [df.iloc[i] for i in range(len(df.index))]
        return batch

    def _rank_topics(self, batches):
        batches.sort()
        return batches

    def _get_unique_combinations(self, batches):
        return list(itertools.combinations(self._rank_topics(batches), 2))

    def _add_unique_combinations(self, unique_combinations, edge_dict):

        for combination in unique_combinations:
            if combination in edge_dict:
                edge_dict[combination] += 1
            else:
                edge_dict[combination] = 1
        return edge_dict
        # calculates the edges and nodes that exist in the list of hashtag lists

    def _get_edge_df(self, batches):

        self.batches = self._get_batches(self.df)
        edge_dict = {}
        source = []
        target = []
        edge_frequency = []

        # execute functions as above looping through each list, finding all unique combinations in each list
        # and adding them to dict object
        for batch in batches:
            edge_dict = self._add_unique_combinations(self._get_unique_combinations(batch), edge_dict)

        # create edge dataframe
        for key, value in edge_dict.items():
            source.append(key[0])
            target.append(key[1])
            edge_frequency.append(value)

        edge_df = pd.DataFrame({'source': source, 'target': target, 'edge_frequency': edge_frequency})
        edge_df.sort_values(by='edge_frequency', ascending=False, inplace=True)
        edge_df.reset_index(drop=True, inplace=True)
        return edge_df

    def _get_node_df(self, edge_df):
        node_df = pd.DataFrame({'id': list(set(list(edge_df['source']) + list(edge_df['target'])))})
        node_df['id_code'] = node_df.index
        return node_df

    def _get_node_id_dic(self, node_df):
        dic_values = [i for i in range(len(node_df['id']))]
        # print(node_df['id'])
        return dict(zip(node_df['id'], dic_values))

    def _get_node_dic(self, node_id_dic):
        return dict((v, k) for k, v in node_id_dic.items())

    def _get_adjacency_dic(self, id, adjacency_frequency):
        return dict(zip(self.node_df[id], self.node_df[adjacency_frequency]))

    def _updated_edge_df(self, edge_df, node_dict):
        edge_df['source_code'] = edge_df['source'].apply(lambda x: node_dict[x])
        edge_df['target_code'] = edge_df['target'].apply(lambda x: node_dict[x])
        return edge_df

    def _extract_edges(self, edge_df):
        tuple_out = []
        for i in range(0, len(edge_df.index)):
            tuple_out.append((edge_df['source_code'][i], edge_df['target_code'][i]))
        return tuple_out

    def _build_graph(self, edge_df, node_df):
        G = nx.Graph()
        G.add_nodes_from(node_df.id_code)
        edge_tuples = self._extract_edges(edge_df)
        for i in edge_tuples:
            G.add_edge(i[0], i[1])
        return G

    def _community_allocation(self, source_val):
        for k, v in self.graph_communities_dict.items():
            if source_val in v:
                return k

    def _get_no_edge_df(self):

        no_edge_df = pd.DataFrame(list(nx.non_edges(self.G)))
        no_edge_df.columns = ['source', 'target']
        no_edge_df['source'] = no_edge_df['source'].apply(lambda x: self.node_dic[x])
        no_edge_df['target'] = no_edge_df['target'].apply(lambda x: self.node_dic[x])
        no_edge_df['source_adjacency'] = no_edge_df['source'].apply(lambda x: self.adjacency_dic[x])
        no_edge_df['target_adjacency'] = no_edge_df['target'].apply(lambda x: self.adjacency_dic[x])
        return no_edge_df

    def fit_transform(self):
        print('Getting edges and nodes..')
        self.edge_df = self._get_edge_df(self.df)
        self.node_df = self._get_node_df(self.edge_df)
        self.node_id_dic = self._get_node_id_dic(self.node_df)
        self.node_dic = self._get_node_dic(self.node_id_dic)
        self.edge_df = self._updated_edge_df(self.edge_df, self.node_id_dic)

        print('Building graph..')
        self.G = self._build_graph(self.edge_df, self.node_df)

        print('Getting graph features..')
        self.graph_adjacencies = dict(self.G.adjacency())
        self.graph_betweeness = nx.betweenness_centrality(self.G)
        self.graph_clustering_coeff = nx.clustering(self.G)
        self.graph_communities = nx.community.greedy_modularity_communities(self.G)

        print('Updating DataFrames..')
        self.node_df['adjacency_frequency'] = self.node_df['id_code'].map(lambda x: len(self.graph_adjacencies[x]))
        self.node_df['betweeness_centrality'] = self.node_df['id_code'].map(lambda x: self.graph_betweeness[x])
        self.node_df['clustering_coefficient'] = self.node_df['id_code'].map(lambda x: self.graph_clustering_coeff[x])
        self.adjacency_dic = self._get_adjacency_dic('id', 'adjacency_frequency')
        self.no_edge_df = self._get_no_edge_df()
        self.graph_communities_dict = {}
        nodes_in_community = [list(i) for i in self.graph_communities]

        for i in nodes_in_community:
            self.graph_communities_dict[nodes_in_community.index(i)] = i
        self.node_df['community'] = self.node_df['id_code'].map(lambda x: self._community_allocation(x))


if __name__ == '__main__':
    # df = pd.DataFrame({'A':[['dog','cat','pig'],['penguin','cat','pig'],['dog','bird','pig']]})
    # print(df)
    #
    # net = NetworkGeneration(df['A'])
    # net.fit_transform()
    # model = BertTransformer()
    # results = model.fit_transform(['This is some text'])

    model_clip = ClipTransformer()
    result_image = model_clip.fit_transform(['this is some text'], transform_type='text')

    model_bert = BertTransformer()
    result_bert = model_bert.fit_transform(['this is some text'])
    #
    model_use = UniversalSentenceTransformer()
    result_use = model_use.fit_transform(['this is some text','this is some more'])
    # USE = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    print('stop')
