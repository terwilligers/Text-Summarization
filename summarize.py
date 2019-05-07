import argparse
import nltk
import numpy as np
import tqdm
import sklearn.metrics
import collections
import math

from rouge_implementation import rouge
from rouge_implementation import process_sentence

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
Returns a summary consisting of purely the first sentence
in a paragraph. A lede 1 baseline.
'''
def summarize_doc_constant1(doc):
   sentences = nltk.sent_tokenize(doc)
   return sentences[0]

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


'''
Create logistic regression model
For all training set docs, assign each sentence in summary a 1, and each sentence
in doc a number between 0 and 1 based on how close to summary sentences it is?
Right now our features are the words, and docs are assigned 0, while summaries are assigned 1
'''
def get_logistic_regression(train_docs,train_summaries, limit, min_vocab_occur=20):

    import sklearn.linear_model
    import sklearn.preprocessing
    import sklearn.pipeline

    model = sklearn.pipeline.Pipeline([('scaler', sklearn.preprocessing.StandardScaler()),
                                       ('model', sklearn.linear_model.LogisticRegressionCV(Cs=5))])

    #create vocab
    word_counts = collections.Counter()
    for doc in train_docs:
        for word in tokenize(doc):
            word_counts.update(word)

    vocab = [w for w, c in word_counts.items() if c >= min_vocab_occur]
    word2idx = {w:idx for idx, w in enumerate(vocab)}

    #Find total number of sentences
    count = 0
    for doc in train_docs:
        for sentence in nltk.sent_tokenize(doc):
            count += 1
    for doc in train_summaries:
        for sentence in nltk.sent_tokenize(doc):
            count += 1

    data_matrix = np.zeros((count, len(vocab)))
    train_labels = np.zeros(count)

    #loop through each sentence and create word vectors
    #currently assigning doc sentences to 0, and sum sentences to 1, but that should change
    offset = 0
    for i,doc in enumerate(train_docs):
        summary = train_summaries[i]
        doc_sentences = nltk.sent_tokenize(doc)
        sum_sentences = nltk.sent_tokenize(summary)

        for s in doc_sentences:
            words = tokenize(s)
            for word in words:
                if word in word2idx:
                    data_matrix[offset, word2idx[word]] += 1
            offset += 1
            #maybe assign some train labels to 1 here, say calculate overlap vocab from sum, and assign 1 if enough overlap?

        #currently assigning only sum sentences to 1
        for s in sum_sentences:
            words = tokenize(s)
            for word in words:
                if word in word2idx:
                    data_matrix[offset, word2idx[word]] += 1
            train_labels[offset] = 1
            offset += 1

    model.fit(data_matrix, train_labels)

    return word2idx, model

'''
Slightly better than baseline summarizer, based on logistic regression model

For each test/val set doc use logistic regression to assign each sentence
to a predict_probability between 0 and 1, then select sentences based on this.

--To start lets just assign 0's and 1's?
for each sentence we assign class, and then sort and take top 4 as summary.

--Problems: Considers sentences across all docs, so all its really doing is picking out
sentences that look like previous sum sentences, but not like previous doc sentences, is this
really what we want?
'''
def summarize_logistic_reg(doc, vocab_lr, model_lr):
    sentences = nltk.sent_tokenize(doc)
    results = []
    summary = ""
    #create word count vectors for each sentence.
    for i,s in enumerate(sentences):
        term_count_vector = np.zeros((1,len(vocab_lr)))
        words = tokenize(s)
        for w in words:
            if w in vocab_lr:
                term_count_vector[0][vocab_lr[w]] += 1

        #gets logistic regression probabilities based on sentence words.
        output = model_lr.predict_proba(term_count_vector)[0]
        results.append(output.tolist() +[i])

    #take top 4 as summary, then sort back into initial order
    results.sort(key=lambda x: x[0])
    top4 = results[:4]
    top4.sort(key=lambda x: x[2])
    for i in range(4):
        if i >= len(sentences):
            break
        summary = summary + ' ' + sentences[top4[i][2]]
    return summary

'''
Uses our rouge implementation to compute summarization metrics
We take the average over every summary
'''
def get_rouge_avg(system_sums, val_sums, n):
    N = len(system_sums)
    total_recall = 0
    total_precision = 0
    total_f1 = 0
    for i,sys_sum in enumerate(system_sums):
        #error handle check
        if len(process_sentence(sys_sum)) < n:
            N -= 1
            continue
        val_sum = val_sums[i]
        (recall, precision, f1) = rouge(process_sentence(sys_sum), process_sentence(val_sum), n)
        total_recall += recall
        total_precision += precision
        if f1:
            total_f1 += f1

    return total_recall/N, total_precision/N, total_f1/N


def main():
    limit_num = 100
    args = parse_args()
    train_docs, train_sums = load_labeled_corpus(args.train_titles, limit = limit_num)
    val_docs, val_sums = load_labeled_corpus(args.val_titles, limit = limit_num)

    ## Constant prediction
    constant1_predictions = np.array([summarize_doc_constant1(v) for v in tqdm.tqdm(val_docs)])

    #better than baseline prediction
    vocab_lr, model_lr =  get_logistic_regression(train_docs, train_sums, limit_num)
    lr_predictions = np.array([summarize_logistic_reg(v, vocab_lr, model_lr) for v in tqdm.tqdm(val_docs)])

    #api prediction
    sumy_predictions = np.array([summarize_sumy(v) for v in tqdm.tqdm(val_docs)])

    print("Rouge-2 metrics (recall, precision, f1):")
    print("baseline:", get_rouge_avg(constant1_predictions, val_sums, 2))
    print("logistic:", get_rouge_avg(lr_predictions, val_sums, 2))
    print("sumy:", get_rouge_avg(sumy_predictions, val_sums, 2))


    # for i in range(3):
    #     print("baseline:")
    #     print(constant1_predictions[i])
    #     print(rouge(process_sentence(constant1_predictions[i]), process_sentence(val_sums[i]), 2))
    #     print()
    #     print("Logistic Regression:")
    #     print(lr_predictions[i])
    #     print(rouge(process_sentence(lr_predictions[i]), process_sentence(val_sums[i]), 2))
    #     print()
    #     print("sumy summary:")
    #     print(sumy_predictions[i])
    #     print(rouge(process_sentence(sumy_predictions[i]), process_sentence(val_sums[i]), 2))
    #     print()
    #     print("actual summary:")
    #     print(val_sums[i])
    #     print()








if __name__ == '__main__':
    main()
