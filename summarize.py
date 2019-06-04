import argparse
import nltk
import numpy as np
import tqdm
import sklearn.metrics
import collections
import math

from gensim.models.keyedvectors import KeyedVectors
from rouge_score import *
from rouge_implementation import rouge
from rouge_implementation import process_sentence
from text_rank_implementation import text_rank

'''sumy api packages'''
from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

'''variety of sumy summarizers'''
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer

'''sumy api settings'''
LANGUAGE = "english"
SENTENCES_COUNT = 3


global _TOKENIZER
_TOKENIZER = nltk.tokenize.casual.TweetTokenizer(
    preserve_case=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t','--train_titles',
                        type=str,
                        default='train_samples',
                        help='Path to training doc titles')
    parser.add_argument('-v','--val_titles',
                        type=str,
                        default='val_samples',
                        help='Path to validation doc titles')

    parser.add_argument('--limit',
                        type=int,
                        default=10,
                        help='Number of documents to summarize')
    parser.add_argument('--word2vec_init',
                        type=int,
                        default=0,
                        help='Whether to compute TextRank similarity scores with word2vec')
    return parser.parse_args()


def tokenize(string):
    '''Given a string, consisting of potentially many sentences, returns
    a lower-cased, tokenized version of that string.
    '''
    global _TOKENIZER
    return _TOKENIZER.tokenize(string)


'''
Load the corpus, based on the structure of our sample files, which is a document
followed by two new lines followed by a summary followed by two new lines...

'''
def load_labeled_corpus(fname, limit):
    docs = []
    summaries = []

    with open(fname, encoding='utf-8') as f:
        doc = True
        count = 0
        for line in tqdm.tqdm(f):
            if count == limit:
                break
            if line == '\n':
                continue
            elif doc:
                docs.append(line.rstrip('\n') + ' ')
                doc = False
            else:
                summaries.append(line.rstrip('\n') + ' ')
                doc = True
                count += 1

    return docs, summaries

'''
Returns a summary using the sumy API
The structure of the sumy API requires writing our document to a temporary file
'''
def summarize_sumy(doc, case):
    summary = ""
    file_doc = open("temp.txt", "w", encoding = 'utf-8')
    file_doc.write(doc)
    file_doc.close()

    parser = PlaintextParser.from_file("temp.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    if case == 1:
        summarizer = LexRankSummarizer(stemmer)
    else:
        summarizer = LsaSummarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)
    for sentence in summarizer(parser.document, SENTENCES_COUNT):
        summary += str(sentence) + ' '

    return summary


'''
Gets metrics from the dataset
'''
def get_metrics(train_docs):
    vocab = set()
    for doc in train_docs:
        for word in tokenize(doc):
            if word not in vocab:
                vocab.add(word)
    return vocab



'''
Uses our rouge implementation to compute summarization metrics
We take the average over every summary
'''
def get_rouge_avg(system_sums, val_sums, method):
    N = len(system_sums)
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    for i,sys_sum in enumerate(system_sums):

        if method != 'l':
        #error handle check

            if len(process_sentence(sys_sum)) < method:
                N -= 1
                continue
            val_sum = val_sums[i]
            (recall, precision, f1) = rouge(process_sentence(sys_sum), process_sentence(val_sum), method = method)
            total_recall += recall
            total_precision += precision
            if f1:
                total_f1 += f1

        if method == 'l':
            val_sum = val_sums[i]
            (recall, precision, f1) = rouge(process_sentence(sys_sum), process_sentence(val_sum), method = method)
            total_recall += recall
            total_precision += precision
            if f1:
                total_f1 += f1

    return 100 * total_recall/N, 100 * total_precision/N, 100 * total_f1/N

'''
Calls an api ROUGE implementation, which we compare our implementation too.
'''
def get_rouge_api(system_sums, val_sums, method):
    N = len(system_sums)
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    for i,sys_sum in enumerate(system_sums):
        if method != 'l':

        #error handle check
            if len(process_sentence(sys_sum)) < method:
                N -= 1
                continue
            val_sum = val_sums[i]
            scores = rouge_n(process_sentence(sys_sum), process_sentence(val_sum), method)
            recall = scores["r"]
            precision = scores["p"]
            f1 = scores["f"]
            total_recall += recall
            total_precision += precision
            if f1:
                total_f1 += f1
        if method == 'l':
            val_sum = val_sums[i]
            scores = rouge_l_summary_level(process_sentence(sys_sum), process_sentence(val_sum))
            recall = scores["r"]
            precision = scores["p"]
            f1 = scores["f"]
            total_recall += recall
            total_precision += precision
            if f1:
                total_f1 += f1

    return 100 * total_recall/N, 100 * total_precision/N, 100 * total_f1/N



'''
Summarization using our better than baseline TextRank implementation
'''
def summarize_text_rank(doc, wv_model=None):
    sentences = nltk.sent_tokenize(doc)
    summary = text_rank(sentences, 3, wv_model=wv_model)
    return ' '.join(summary)

'''
Lede-n baseline
'''
def summarize_doc_constant(doc, n):
   sentences = nltk.sent_tokenize(doc)
   result = ""
   for i in range(n):
       if i >= len(sentences):
           break
       result = result + ' ' + sentences[i]
   return result


def main():
    args = parse_args()
    limit_num = args.limit
    train_docs, train_sums = load_labeled_corpus(args.train_titles, limit = limit_num)
    val_docs, val_sums = load_labeled_corpus(args.val_titles, limit = limit_num)

    # print some information about how many docs we loaded and our vocab size
    val_vocab = get_metrics(val_docs)
    print('Loaded {0} training docs and {1} validation/test docs.'.format(len(train_docs), len(val_docs)))
    print('There are {} unique word types in our validation vocab overall.'.format(len(val_vocab)))

    ## Constant prediction, using lede-3
    constant_n = 3
    constant_predictions = np.array([summarize_doc_constant(v, constant_n) for v in tqdm.tqdm(val_docs)])

    #api prediction
    sumy_lexrank_predictions = np.array([summarize_sumy(v, 1) for v in tqdm.tqdm(val_docs)])
    sumy_lsa_predictions = np.array([summarize_sumy(v, 2) for v in tqdm.tqdm(val_docs)])

    #TextRank implementation prediction, if word2vec_init == 1, then we compute similarities with word2vec
    if args.word2vec_init == 0:
        text_rank_pred = np.array([summarize_text_rank(v) for v in tqdm.tqdm(val_docs)])
    else:
        wv_model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True, limit=100000)
        text_rank_pred_w2vec = np.array([summarize_text_rank(v, wv_model=wv_model) for v in tqdm.tqdm(val_docs)])

    #displays rouge metrics
    for n in list(range(1,3)) + ['l']:
        print("Rouge-{} metrics (recall, precision, f1):".format(n))

        print("lede-{} baseline:".format(constant_n), get_rouge_avg(constant_predictions, val_sums, n))
        print("PyRouge:", get_rouge_api(constant_predictions, val_sums, n))
        print()

        if args.word2vec_init == 0:
            print("TextRank:", get_rouge_avg(text_rank_pred, val_sums, n))
            print("PyRouge:", get_rouge_api(text_rank_pred, val_sums, n))
        else:
            print("TextRank with word2vec similarities:", get_rouge_avg(text_rank_pred_w2vec, val_sums, n))
            print("PyRouge:", get_rouge_api(text_rank_pred_w2vec, val_sums, n))
        print()


        print("LexRank sumy:", get_rouge_avg(sumy_lexrank_predictions, val_sums, n))
        print("PyRouge:", get_rouge_api(sumy_lexrank_predictions, val_sums, n))
        print()

        print("LSA sumy:", get_rouge_avg(sumy_lsa_predictions, val_sums, n))
        print("PyRouge sumy:", get_rouge_api(sumy_lsa_predictions, val_sums, n))
        print()










if __name__ == '__main__':
    main()
