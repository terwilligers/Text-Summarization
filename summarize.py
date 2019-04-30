from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import argparse
import nltk
import numpy as np
import tqdm
import sklearn.metrics
import collections
import math

'''sumy api packages'''
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

'''sumy api settings'''
LANGUAGE = "english"
SENTENCES_COUNT = 3


global _TOKENIZER
_TOKENIZER = nltk.tokenize.casual.TweetTokenizer(
    preserve_case=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_titles',
                        type=str,
                        default='train_samples',
                        help='Path to training doc titles')
    parser.add_argument('--val_titles',
                        type=str,
                        default='val_samples',
                        help='Path to validation doc titles')
    return parser.parse_args()


def tokenize(string):
    '''Given a string, consisting of potentially many sentences, returns
    a lower-cased, tokenized version of that string.
    '''
    global _TOKENIZER
    return _TOKENIZER.tokenize(string)


'''
Load the corpus, based on the structure of our sample files
'''
def load_labeled_corpus(fname, limit):
    docs = []
    summaries = []

    with open(fname) as f:
        doc = True
        count = 0
        for line in tqdm.tqdm(f):
            if count == limit:
                break
            if line == '\n':
                continue
            elif doc:
                docs.append(line.rstrip('\n'))
                doc = False
            else:
                summaries.append(line.rstrip('\n'))
                doc = True
                count += 1

    return docs, summaries

'''
Returns a summary consisting of purely the first sentence (everything before a period)
in a paragraph.
'''
def summarize_doc_constant1(doc):
   sentences = doc.split('.')
   return sentences[0] + '.'

'''
Returns a summary using the sumy API
'''
def summarize_sumy(doc):
    summary = ""
    file_doc = open("temp.txt", "w")
    file_doc.write(doc)
    file_doc.close()

    parser = PlaintextParser.from_file("temp.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        summary += str(sentence) + ' '

    return summary


def main():
    args = parse_args()
    train_docs, train_sums = load_labeled_corpus(args.train_titles, limit = 100)
    val_docs, val_sums = load_labeled_corpus(args.val_titles, limit = 100)

    ## Constant prediction
    constant1_predictions = np.array([summarize_doc_constant1(v) for v in val_docs])

    sumy_predictions = np.array([summarize_sumy(v) for v in tqdm.tqdm(val_docs)])

    for i in range(2):
        print("baseline:")
        print(constant1_predictions[i])
        print()
        print("sumy summary:")
        print(sumy_predictions[i])
        print()
        print("actual summary:")
        print(val_sums[i])
        print()






if __name__ == '__main__':
    main()
