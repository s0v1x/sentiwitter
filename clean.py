import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from emot.emo_unicode import UNICODE_EMO, EMOTICONS
import contractions
import numpy as np

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def remove_punctuation(s):
    return ''.join(c for c in s if c not in '!"#$%&\'()*+,-.:;<=>?@[\\]^_`{|}~' )
    
def preprocess(txt):
  
  lemmmatizer=WordNetLemmatizer()
  word_tokens = [word.lower() for word in word_tokenize(txt)]
  words = [lemmmatizer.lemmatize(word.lower()) for word in word_tokens if(not word.lower() in set(stopwords.words('english')) and  word.isalpha())] 
  words = [convert_slang(word.lower()) for word in words] 
  return words

def convert_emoticons(text):
  for emot in EMOTICONS:
    text = re.sub(u'('+emot+')', "_".join(EMOTICONS[emot].replace(",","").split()), text)
  return text

from slang import *
def convert_slang(token):
  try :
    return slang[token]
  except:
    return token

def clean_tweet(tweet):
      tweet = re.sub(r"http\S+", " ", tweet)
      tweet = re.sub(r"@[\w]*", " ", tweet)
      tweet = contractions.fix(tweet)
      tweet = convert_emoticons(tweet)
      tweet = re.sub(r"[^a-zA-Z#]", " ", tweet)
      tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",tweet).split())
      tweet = remove_punctuation(tweet)
      tweet = preprocess(tweet)
      return tweet


def docvec(corpus,model):
    v=np.zeros(100)
    for w in corpus:
        v+=model.wv[w]
        v/=len(corpus)
    return v