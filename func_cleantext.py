import re  # For preprocessing
import pandas as pd  # For data handling
import numpy as np
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing
from datasets import load_dataset


nlp = spacy.load('en_core_web_lg') # lemmatization object



bad_lemmas = ['', "'", 'ad', 'ah', 'ai', 'al', 'ap', 'av', 'b', 
              'bc', 'bi', 'bs', 'c', 'ck', 'co', 'cr', 'd', 'da', 
              'dc', 'de', 'du', 'e', 'ed', 'el', 'en', 'es', 'et', 
              'ex', 'f', 'fe', 'fi', 'g', 'ga', 'h', 'ha', 'hi', 
              'hq', 'hr', 'ia', 'ii', 'iq', 'j', 'jo', 'jp', 'k', 
              'kd', 'ki', 'ku', 'ky', 'l', 'la', 'le', 'li', 'll', 
              'lo', 'lt', 'lv', 'lx', 'm', 'md', 'mm', 'mo', 'mp', 
              'mx', 'n', 'na', 'nd', 'nt', 'nv', 'o', 'oh', 'ok', 
              'om', 'op', 'os', 'ou', 'p', 'pc', 'pi', 'pm', 'ps', 
              'q', 'r', 'rc', 'rd', 'ro', 'rs', 's', 'se', 'si', 
              'st', 't', 'th', 'tk', 'to', 'tr', 'tu', 'tv', 'tx', 
              'u', 'uc', 'um', 'un', 'ut', 'ux', 'uz', 'v', 'va', 
              'vc', 've', 'vp', 'vt', 'w', 'wc', 'wi', 'wj', 'x', 
              'y', 'yr', 'z']

mapping_dict = {'Vox': 'vxo',
                 'Vice News': 'vnw',
                 'Reuters': 'rtr',
                 'Vice': 'vce',
                 'TMZ': 'tmz',
                 'Hyperallergic': 'hpr',
                 'Business Insider': 'bsi',
                 'TechCrunch': 'tcr',
                 'Axios': 'axs',
                 'Refinery 29': 'rfn',
                 'The Verge': 'vrg',
                 'Mashable': 'msh',
                 'People': 'ppl',
                 'Economist': 'ecn',
                 'CNN': 'cnn',
                 'Gizmodo': 'gzd',
                 'New Yorker': 'nwy',
                 'Wired': 'wrd',
                 'CNBC': 'cnb',
                 'New Republic': 'nwr',
                 'Fox News': 'fxn',
                 'The Hill': 'hll',
                 'Politico': 'plt',
                 'Buzzfeed News': 'bzn',
                 'The New York Times': 'nyt',
                 'Washington Post': 'wpt'}



# ------------------- Functions ------------------ #

def clean_articles(examples, bad_lemmas=bad_lemmas, mapping_dict=mapping_dict):
    '''
    examples : dataset or batch of a dataset 
    bad_lemmas :  lemmas to exclude from the cleaned article
    mapping_dict :  mapping of publication titles to target_word to replace "immigr"

    return
    ------------
    dict of clean articles, replaces examples['article'] column
    '''
    
    clean_articles = []
    for article, publication in zip(examples['article'], examples['publication']):
        # ------- Clean the text
        article = re.sub("[^A-Za-z']+", ' ', article.lower()) # lowercase and remove non alphanumeric chars
        article = nlp(article) # convert to nlp object    
        article = [token.lemma_ for token in article if (not token.is_stop) & (token.lemma_ not in bad_lemmas)] # # lemmatize and drop bad lemmas
        article = ' '.join(article) # join list of lemmas

        # ----- replace immigr
        target_word = mapping_dict[publication] # target word to replace immigr
        article = re.sub(r'\bimmigr\w+\b', target_word, article) # replace immigr with target_word

        clean_articles.append(article) # append final result to the clean list
            
    
    return {'article': clean_articles} # return as a dict for dataset batch compatibility

