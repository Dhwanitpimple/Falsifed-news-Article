import pandas as pd
from feature_engineering import hand_features
from bs4 import BeautifulSoup
import requests
import time
import numpy as np
import joblib
from summarizer import summarize
st=joblib.load("saved_mo/model_mnb.pkl")
title = input()


def rele(a):
	url = "https://news.google.com/search?q="+a
	k=[]
	try:
		page=requests.get(url)
	except Exception as e:
		error_type, error_obj, error_info = sys.exc_info()
		print ('ERROR FOR LINK:',url)
		print (error_type, 'Line:', error_info.tb_lineno) 
                                          
	time.sleep(2)   
	soup=BeautifulSoup(page.text,'html.parser')
	Lin = soup.find_all("h3")
	for i in Lin[1:4]:
		k.append(i.text.strip())
	return k
    

test_body=[]
test_head=rele(title)
test=title
for i in range(3):
    test_body.append(test)
print(title)
print("Titles: "+str(test_head[:]))
print("Title:   "+test_body[2])
test_body_fea= test_body[:]
test_head_fea= test_head[:]
test_hand_features_mat= np.zeros((len(test_head_fea),1))
test_hand_features_mat= hand_features(test_head_fea, test_body_fea)
print(type(test_hand_features_mat),test_hand_features_mat.shape)
test_final_fea= test_hand_features_mat
prediction = st.predict(test_final_fea)
print(prediction)