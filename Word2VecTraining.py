import re  # For preprocessing
import pandas as pd  # For data handling
import numpy as np
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing
from datasets import load_dataset


def parse_articles(examples):
	'''
	Break down each article into a list of words/strings
	examples: huggingface dataset object
	'''
	parsed_articles = [article.split(' ') for article in examples['article']]
	return {'parsed_article':parsed_articles}


clean_data = load_dataset('csv', data_files='/Users/jd3000/Desktop/neuromatch/all-the-news-2-1_clean.csv')
clean_data = clean_data['train'] # select training set (all the data here)
clean_data = clean_data.filter(lambda x: x['article'] is not None) # drop empty articles
parsed_data = clean_data.map(lambda examples: parse_articles(examples), batched=True) # parse into separate words




t0 = time.time()

model = Word2Vec(parsed_data['parsed_article'])

t1 = time.time()
total = t1-t0
print(total)


# model.save("word2vec.model")
