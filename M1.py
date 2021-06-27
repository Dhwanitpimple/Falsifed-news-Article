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

url="https://www.politifact.com/factchecks/2020/dec/07/viral-image/no-danish-study-didnt-prove-wearing-masks-ineffect/"
try:        
    page=requests.get(url)
        
except Exception as e:    
    error_type, error_obj, error_info = sys.exc_info()
    print ('ERROR FOR LINK:',url)                     
    print (error_type, 'Line:', error_info.tb_lineno) 
                                          
time.sleep(2)   
soup=BeautifulSoup(page.text,'html.parser')
Lin = soup.find("h2",attrs={'class':'c-title c-title--subline'}).text.strip()
ar=soup.find("article",attrs={'class':'m-textblock'})
arti=ar.find_all("p")
ar2=soup.find("span",attrs={'class':'m-author__date'}).text.strip()
print(ar2)
print("\033[1m"+Lin+"\033[0m")

for a in arti:
  print(a.text.strip())
test_list = []
ws = WordNetLemmatizer()
title = re.sub("[^a-zA-Z]"," ",Lin)
title = title.lower()
title = title.split()
title = [ws.lemmatize(word) for word in title if word not in stopwords.words('english')]
title = " ".join(title)
test_list.append(title)
print(test_list)
vocab_size = 10000
one_hot_test = [one_hot(i,vocab_size) for i in test_list]
embed_test = pad_sequences(one_hot_test,maxlen = 356,padding = 'pre')
ypred = imported2.predict(embed_test)
pred = (ypred>0.5).astype('int')
test_o = pd.DataFrame(data = pred,columns = ['label'])
print(test_o.head())