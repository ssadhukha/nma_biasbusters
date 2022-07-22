import re, spacy, time, multiprocessing
import pandas as pd
import numpy as np
from collections import defaultdict
from datasets import load_dataset
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec

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
clean_data = clean_data.map(lambda examples: parse_articles(examples), batched=True) # parse into separate words


# ---------------- Train the model ----------------- #
n_cores = multiprocessing.cpu_count()


class callback(CallbackAny2Vec):
    '''Callback to print loss after each epoch.'''

    def __init__(self):
        self.epoch = 0
        self.loss_to_be_subed = 0

    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_now = loss - self.loss_to_be_subed
        self.loss_to_be_subed = loss
        print('Loss after epoch {}: {}'.format(self.epoch, loss_now))
        self.epoch += 1

        

t0 = time.time()        
model = Word2Vec(clean_data['parsed_article'],
                 vector_size=300,
                 window=5,
                 min_count=100,
                 negative=20,
                 sample=1e-5,
                 sg=1,
                 compute_loss=True,
                 workers=n_cores-1,
                 compute_loss=True, 
                 callbacks=[callback()])
 

t1 = time.time()
total = t1-t0
print(total)

# model.save('./model/v2.model')
