import re
import pandas as pd
import nltk
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import urllib.request,sys,time
from bs4 import BeautifulSoup
import requests
import pandas as pd
nltk.download('stopwords')
nltk.download('wordnet')

imported2 = tf.keras.models.load_model('saved_mo/my_mo')
test_list = []
ws = WordNetLemmatizer()
title = re.sub("[^a-zA-Z]"," ","Param Bir Singh letter: No question of replacing Anil Deshmukh, says NCP leader Jayant Patil")
title = title.lower()
title = title.split()
title = [ws.lemmatize(word) for word in title if word not in stopwords.words('english')]
title = " ".join(title)
test_list.append(title)
print(test_list)
vocab_size = 10000
one_hot_test = [one_hot(i,vocab_size) for i in test_list]
embed_test = pad_sequences(one_hot_test,maxlen = 356,padding = 'pre')
print(embed_test)
ypred = imported2.predict(embed_test)
pred = (ypred>0.5).astype('int')
test_o = pd.DataFrame(data = pred,columns = ['label'])
print(test_o.head())