Text-Summarization Code:

The main program for our project is called summarize.py, and can be run just as "python summarize.py" with the default parameters. This program takes document and summary pairs from two files, "train_samples" and "val_samples", which contain Wikihow article segments. These consists of documents and summaries on individual lines, each separated by two new lines. It then runs multiple summarization models on the validation dataset to see which produces better ROUGE scores.

In addition to the required data files, the program also requires that the following files appear in the directory: rouge_implementation.py, rouge_score.py, and text_rank_implementation.py. The rouge files contain functions to compute validation metrics using both our own implementation and an api. The text_rank file contains our better than baseline implementation of the textRank algorithm.

The program can be run with a variety of command line arguments. By default it runs the algorithms on all 1000 documents in the validation set, but with the --limit command you can specify to limit the program to running on the first-n documents such as "--limit 100" for 100 documents. Moreover, you can also run the program with "--word2vec_init 1" to run textRank with word2vec similarity averages. This actually hurts our rouge scores, but it's an interesting experiment.

Finally, while running the program we call an external api that requires writing the data to a temp.txt file, which will appear in the directory when ran. This can be ignored.
