import re  # For preprocessing
import pandas as pd  # For data handling
import numpy as np
from time import time  # To time our operations
from collections import defaultdict  # For word frequency
import spacy  # For preprocessing
from datasets import load_dataset


nlp = spacy.load('en_core_web_lg') # lemmatization object

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

# ------------------- Functions ------------------ #
def _clean_string(text, bad_lemmas=bad_lemmas, min_words=0):
    '''
    text: str,  article text
    bad_lemmas : lemmas to exclude from outpur
    min_words :  only return text with min_words output words

    ------------
    return : cleaned text string
    '''


    text = re.sub("[^A-Za-z']+", ' ', text.lower()) # lowercase and remove non alphanumeric chars
    text = nlp(text) # convert to nlp object
    text = [token.lemma_ for token in text if (not token.is_stop) & (token.lemma_ not in bad_lemmas)] # # lemmatize and drop bad lemmas

    # text = [token.lemma_ for token in text if not token.is_stop] # lemmatize and remove stop words
    # text = [token.lemma_ for token in text if (not token.is_stop) &  (not token.ent_type_)] # lemmatize and remove stop words amd named entities

    if len(text) > min_words: # only return sentences with more than min_words words
        return ' '.join(text) # join list of lemmas and return

def _replace_immigr(text, target_word):
    '''
    Replace all strings containing 'immigr' with the target_word.
    e.g. immigration --> target_word
         immigrant --> target_word

    ------------
    return : str
    '''
    return re.sub(r'\bimmigr\w+\b', target_word, text, flags=re.IGNORECASE)



def process_data(row, mapping_dict=mapping_dict, out_type='str'):
    '''
    row : dataframe or dataset row, acceced when using .map() or .apply()

    mapping_dict : dict of mapping of immigr to a pseudoword for each pulication
                   e.g. {'vox':'vxo', 'New York Times':'nty'}

    out_type : 'str' returns the clean text as a string. Use to apply function on
                a pandas.DataFrame
                e.g. df['out] = df.apply(process_data, axis=1)

               'dict' returns a dictionary. Use when working with huggingface datasets
                e.g. clean_data = dataset.map(lambda x: process_data(x, out_type='dict'))
    '''

    clean = _clean_string(row['article']) # clean the article
    pseudoword = mapping_dict[row['publication']] # what to replace 'immigr' words with
    clean = _replace_immigr(clean, pseudoword) # replace immigr with targ_word

    if out_type == 'str':
        return clean
    elif out_type == 'dict':
        return {'clean_article' : clean}
