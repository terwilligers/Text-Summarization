'''
An implementation of the TextRank algorithm.
Generally this consists of:
-constructing a weighted graph of sentences
-applying the weighted PageRank algorithm to this graph
-selecting the top-n ranked sentences

Source for algorithmic description: https://www.aclweb.org/anthology/P04-3020
'''

import nltk
import math

from gensim.models.keyedvectors import KeyedVectors

global _TOKENIZER
_TOKENIZER = nltk.tokenize.casual.TweetTokenizer(
    preserve_case=False)

'''
Class to represent a sentence in our graph
'''
class Sentence:
    def __init__(self, id, s):
        self.sent_string = s
        self.words = self.tokenize(s)
        self.neighbors = {}
        self.id = id
        self.p_rank = None

    def tokenize(self, string):
        global _TOKENIZER
        return _TOKENIZER.tokenize(string)

    def addNeighbor(self, s_id, weight):
        self.neighbors[s_id] = weight

    def hasNeighbor(self, s_id):
        return s_id in self.neighbors

    def getNeighbors(self):
        return self.neighbors

    def getWords(self):
        return self.words

    def getId(self):
        return self.id

    def getSentString(self):
        return self.sent_string

    def setRank(self, p_rank):
        self.p_rank = p_rank

    def getRank(self):
        return self.p_rank


'''
Class to represent a undirected and weighted similarity graph
'''
class Graph:
    def __init__(self):
        self.sentences = {}

    def addSentence(self, id, s):
        self.sentences[id] = Sentence(id,s)

    def getSentence(self, id):
        return self.sentences[id]

    def addEdge(self, s1, s1_id, s2, s2_id, weight):
        if s1_id not in self.sentences:
            self.addSentence(s1_id, s1)
        if s2_id not in self.sentences:
            self.addSentence(s2_id, s2)
        s1.addNeighbor(s2_id, weight)
        s2.addNeighbor(s1_id, weight)

    def getSentences(self):
        return self.sentences

    def hasEdge(self, s1, s2_id):
        return s1.hasNeighbor(s2_id)

    '''
    outputs a list of sentences sorted by rank
    '''
    def sortByRank(self):
        id_list = []
        for id, s in self.sentences.items():
            id_list.append((id, s.getRank()))
        id_list.sort(key=lambda x: x[1])
        id_list.reverse()
        return id_list


'''
Constructs a sentence graph based on similarity scores
'''
def construct_graph(sentences, wv_model=None):
    #construct graph and sentences
    sim_graph = Graph()
    for i,s in enumerate(sentences):
        sim_graph.addSentence(i,s)

    #construct edges
    sentences = sim_graph.getSentences()
    for id_1, s1 in sentences.items():
        for id_2, s2 in sentences.items():
            #check if already checked this pair
            if id_1 == id_2 or s1.hasNeighbor(id_2):
                continue
            if wv_model:
                similarity = get_word2vec_similarity(s1, s2, wv_model)
            else:
                similarity = get_similarity(s1, s2)
            sim_graph.addEdge(s1, id_1, s2, id_2, similarity)

    return sim_graph


'''
Computes the similarity score between 2 sentences using word overlap
Similarity(s1, s2) = number of words that appear in both s1 and s2 / (log(|s1|) + log(|s2|))
'''
def get_similarity(s1, s2):
    words1 = s1.getWords()
    words2 = s2.getWords()
    words2_set = set(words2)
    overlap = 0

    for w in words1:
        if w in words2_set:
            overlap += 1
    if math.log(len(words1)) + math.log(len(words2)) == 0:
        return overlap
    return overlap / (math.log(len(words1)) + math.log(len(words2)))

'''
Computes the similarity score between 2 sentences using average word2vec similarity
'''
def get_word2vec_similarity(s1, s2, wv_model):
    default = .01
    words1 = [w for w in s1.getWords() if w in wv_model.vocab]
    words2 = [w for w in s2.getWords() if w in wv_model.vocab]
    if len(words1) == 0 or len(words2) == 0:
        return default
    sim = wv_model.n_similarity(words1, words2)
    return sim


'''
Runs the page_rank graph algorithm on our weighted sentence graph.
It iterates until convergence within a given threshold

PageRank(node v_i, param d) = (1-d) + d * score, where score =
    summation over every incoming node v_j of:
        w_ji * PageRank(node v_j) / summation over every outgoing node v_k from v_j of w_kj
'''
def page_rank(graph, d=.85):
    threshold = .01
    #assign initial arbitrary values
    sentences = graph.getSentences()
    for id, s in sentences.items():
        s.setRank(.5)

    #loop until convergence
    converging = True
    while converging:
        converging = False
        #update page_rank of each node
        for id_i, s_i in sentences.items():
            score = 0
            #summation over every incoming node s_j
            for id_j, w_ij in s_i.getNeighbors().items():
                s_j = graph.getSentences()[id_j]
                #summation over every outgoing node s_k from s_j of w_jk
                inner_sum = 0
                for id_k, w_jk in s_j.getNeighbors().items():
                    inner_sum += w_jk
                if inner_sum != 0:
                    score += (w_ij*(s_j.getRank()/inner_sum))

            #check if outside of convergence threshold
            new_rank = 1-d+d*score
            old_rank = s_i.getRank()
            if abs(new_rank - old_rank) > threshold:
                converging = True
            s_i.setRank(1-d+d*score)

    return

'''
Selects the top-n sentences from an output of ranked sentences
'''
def select_top_n(graph, n):
    sorted_sents = graph.sortByRank()
    output = []
    for i in range(n):
        if i >= len(sorted_sents):
            break
        sentence = graph.getSentence(sorted_sents[i][0])
        sent_string = sentence.getSentString()
        output.append(sent_string)
    return output

'''
TextRank algorithm
Takes as input a list of sentences and the summary length, and outputs a summary
'''
def text_rank(sentences, limit, wv_model=None):
    if wv_model:
        sim_graph = construct_graph(sentences, wv_model=wv_model)
    else:
        sim_graph = construct_graph(sentences)
    page_rank(sim_graph)
    top_n = select_top_n(sim_graph, limit)
    return top_n


def main():

    input = ['This is it, one graph to rule them all.',
                 'One graph to find them',
                 'One graph to bring them all and in the darkness bind them',
                 'What is this now.',
                 'Are we really out here']

    input2 = ['This','what']
    wv_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, limit=100000)
    print(text_rank(input, 3, wv_model=wv_model))

if __name__ == '__main__':
    main()
