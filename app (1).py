from flask import Flask,render_template,request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import shutil
import io
import sys
import math
import time
import random
import requests
import nltk
import collections
import numpy as np
import pandas as pd
from newspaper import Article
from os import walk
from datetime import datetime as dew
from bs4 import BeautifulSoup
import requests
import urllib.request,sys,time
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from feature_engineering import hand_features
import joblib
import datetime
from summarizer import summarize
from GoogleNews import GoogleNews
from nltk.tokenize import word_tokenize
from csv import writer
from htmldate import find_date
from flask_sqlalchemy import SQLAlchemy
from flask import (
    Blueprint, flash, g, redirect, render_template, request, session, url_for
)
import sqlite3
from whois import whois
from csv import writer
today = datetime.date.today()
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
st=joblib.load("saved_mo/model_mnb.pkl")
imported2 = tf.keras.models.load_model('saved_mo/my_mo')
src_dir = os.getcwd()
us_id=0
class user(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80))
    email = db.Column(db.String(120))
    password = db.Column(db.String(80))
def rele2(a):
	googlenews = GoogleNews()
	googlenews.get_news(a)
	k=[]
	l=[]
	m=[]
	#for i in range(len(googlenews.result())):
	for i in range(10):
		k.append(googlenews.results()[i]["title"])
		#if i<5:
		try:
			responses=requests.get("http://"+googlenews.results()[i]['link'])
			l.append(responses.url)
			print(l[i])
		except:
			l.append("can't find")
			print(l[i])
	googlenews.clear()
	m.append(k)
	m.append(l)
	return m
def sim(X,Y):     
    X_list = word_tokenize(X) 
    Y_list = word_tokenize(Y)
    
    # sw contains the list of stopwords
    sw = stopwords.words('english') 
    l1 =[];l2 =[]
  
	# remove stop words from the string
    X_set = {w for w in X_list if not w in sw} 
    Y_set = {w for w in Y_list if not w in sw}
  
    # form a set containing keywords of both strings 
    rvector = X_set.union(Y_set) 
    for w in rvector:
        if w in X_set: l1.append(1) # create a vector
        else: l1.append(0)
        if w in Y_set: l2.append(1)
        else: l2.append(0)
    c = 0
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine

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
	for i in Lin[0:4]:
		k.append(i.text.strip())
	return k

def af(X,c):
	pe=pd.read_csv("Datasets/nee.csv")
	flag=0
	for i in list(pe.Headlines):
		if flag==0:
			u=sim(X,i)
			print(u)
		if u>=0.8:
			flag=1
	if flag==0:
		l=[]
		l.append(X)
		l.append(c)
		with open('Datasets/nee.csv','a') as f_object:
			writer_object = writer(f_object)
			writer_object.writerow(l)
def getS(a,o):
	test_body=[]
	test_head=a
	test=o
	for i in range(8):
		test_body.append(test)
	print("Titles: "+str(test_head[:]))
	print("Summary:   "+str(test_body[2]))
	test_body_fea= test_body[:]
	test_head_fea= test_head[:]
	test_hand_features_mat= np.zeros((len(test_head_fea),1))
	test_hand_features_mat= hand_features(test_head_fea, test_body_fea)
	print(type(test_hand_features_mat),test_hand_features_mat.shape)
	test_final_fea= test_hand_features_mat
	prediction = st.predict(test_final_fea)
	return prediction
def simmi(a,b):
    if a==b:
        return 1
    else:
        return 0
def chec(a):
    print(a)
    print("checking trusted"+str(us_id)+".csv")
    pe=pd.read_csv("user_files/trusted"+str(us_id)+".csv")
    flag=0
    for i in list(pe.source):
        if flag==0:
            u=simmi(a,i)
        if u==1:
            flag=1
    if flag==1:
        return 1
    else:
        return 0

def getL(a):
    ws = WordNetLemmatizer()
    #title = re.sub("[^a-zA-Z]"," ",a)
    test_list=[]
    title=a
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
    return pred

@app.route('/')
def loglog():
	return render_template("login.html")

@app.route('/loggin',methods=["GET", "POST"])
def login():
    if request.method == "POST":
        uname = request.form["uname"]
        passw = request.form["passw"]
        global us_id
        login = user.query.filter_by(username=uname, password=passw).first()
        if login is not None:
            us_id=login.id
            return redirect(url_for("ops"))
    return redirect(url_for("loglog"))

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        uname = request.form['uname']
        mail = request.form['mail']
        passw = request.form['passw']
        dest_dir = src_dir+"/user_files"
        
        register = user(username = uname, email = mail, password = passw)
        db.session.add(register)
        db.session.commit()
        conn=sqlite3.connect('database.db')
        dbcc=conn.cursor()
        dbcc.execute('select max(id) from user')
        datadb=dbcc.fetchall()
        src_file = os.path.join(src_dir, 'trusted.csv')
        shutil.copy(src_file,dest_dir)
        iid=datadb[0][0]
        dst_file = os.path.join(dest_dir,'trusted.csv')
        new_dst_file_name = os.path.join(dest_dir, 'trusted'+str(iid)+'.csv')
        os.rename(dst_file, new_dst_file_name)
        return redirect(url_for("login"))
    return render_template(url_for("login"))

@app.route('/index')
def index():
	return render_template("index.html")

@app.route('/getT',methods=['GET','POST'])
def getT():
	if request.method == 'POST':
		print(request.form['Titl'])
		val=None
		ti=request.form['Titl']
		arew=ti
		ba=request.form['bodyn']
		print(ba)
		if ba!='':
			ba2=str(summarize(ti,ba))
		else:
			ba2=ti
		aat=rele2(arew)
		aa=aat[0]
		pq=[]
		links=aat[1]
		ab=getS(aa,ba2)
		ac=getL(ti)
		diff=0
		pr=[]
		today = datetime.date.today()
		date1=datetime.date(int(today.year), int(today.month), int(today.day))
		if ac==1:
			fe="The headline may be falsified"
		if ac==0:
			fe="True"
		dt2=0
		for i in range(len(links)):
				nam=whois(links[i]).domain_name
				if isinstance(nam,list):
					nuy= nam[0].lower()
				elif nam!=None:
					nuy=nam.lower()
				if nam!=None:
					sru=chec(nuy)
					pr.append(nuy)
				else:
					sru=0
					pr.append('unknown')
				if sru==1:
					pq.append("Trusted")
				else:
					pq.append("Not Trusted")
		if request.form['drt']!='':
			dt=dew.strptime(request.form['drt'],'%Y-%m-%d')
			dt2=dt.strftime('%d/%m/%y')
			date2=datetime.date(int(dt.year), int(dt.month), int(dt.day))
			diff=(date1-date2)
		au=request.form['sr']
		af(ti,ac[0][0])
		return render_template("index.html",e=ti,f=val,g=dt2,h=au,l=aa,j=ab,k=fe,ll=str(diff),oo=pq,arli=links,arso=pr)

@app.route('/getURL',methods=['GET','POST'])
def getURL():
	if request.method == 'POST':
		print(request.form['url'])
		print(request.form['date'])
		today = datetime.date.today()
		print(today.year)
		date1=datetime.date(int(today.year), int(today.month), int(today.day))
		val=request.form['url']
		ar=Article(val)
		diff=0
		dt=None
		try:
			ar.download()
			ar.parse()
			ar.nlp()
			ti=ar.title
			arew=ti
			aat=rele2(arew)
			aa=aat[0]
			links=aat[1]
			pq=[]
			pr=[]
			#print(ab)
			dt2=0
			if request.form['date']!='':
				dt=dew.strptime(request.form['date'],'%Y-%m-%d')
				dt2=dt.strftime('%d/%m/%y')
			else:
				dt=dew.strptime(find_date(val),'%Y-%m-%d')
				if dt!=None:
					dt2=dt.strftime('%d/%m/%y')
			if dt!=None:
				date2=datetime.date(int(dt.year), int(dt.month), int(dt.day))
				diff=(date1-date2)

			print(diff)
			nam=whois(val).domain_name
			print(nam)
			if isinstance(nam,list):
				au=nam[0].lower()
			elif nam!=None:
				au=nam.lower()
			else:
				au="google.com"
			srr=chec(au)
			print("Source: "+str(srr))
			if srr==1:
				srg="Trusted"
			else:
				srg="Not Trusted"
			ab=getS(aa,ar.summary)

			for i in range(len(links)):
				nam=whois(links[i]).domain_name
				if isinstance(nam,list):
					nuy= nam[0].lower()
				elif nam!=None:
					nuy=nam.lower()
				if nam!=None:
					sru=chec(nuy)
					pr.append(nuy)
				else:
					sru=0
					pr.append('unknown')
				if sru==1:
					pq.append("Trusted")
				else:
					pq.append("Not Trusted")
			ac=getL(ti)
			if ac==1:
				fe="The headline may be falsified"
			if ac==0:
				fe="True"
			af(ti,ac[0][0])
			return render_template("index.html",e=ar.title,f=val,g=dt2,h=au,l=aa,j=ab,k=fe,ll=str(diff),pp=srg,oo=pq,arli=links,arso=pr)
		except:
			return render_template("index.html",e="This isn't a valid URL, try entering info manually",f="",g="",h="")


@app.route('/getURL/Res',methods=['GET','POST'])
def Res():
	if request.method == 'POST':
		print("We're here")

		return render_template("Res.html")

@app.route("/ops", methods=["GET", "POST"])
def ops():
	return render_template("ops.html")

@app.route("/sour", methods=["GET", "POST"])
def sour():
	pe=pd.read_csv("user_files/trusted"+str(us_id)+".csv")
	a=list(pe.source)
	if request.method == 'POST':
		ow=int(request.form['lol'])
		a.pop(ow)
		if(a[0]!="source"):
			a.insert(0,"source")
		with open("user_files/trusted"+str(us_id)+".csv","w") as writeFile:
			writ = writer(writeFile)
			for iu in a:
				writ.writerow([iu])
		return render_template("sour.html",soe=a[1:],ler=len(a[1:]))

	return render_template("sour.html",soe=a,ler=len(a))

@app.route("/ad", methods=["GET", "POST"])
def ad():
	if request.method == 'POST':
		li=request.form['ure']
		nam=whois(li).domain_name

		if isinstance(nam,list):
			name= nam[0].lower()
			print(name)
		else:
			name=nam.lower()
			print(name)
		pe=pd.read_csv("user_files/trusted"+str(us_id)+".csv")
		flag=0
		for i in list(pe.source):
			if flag==0:
				u=sim(name,i)
			if u==1:
				flag=1
		if flag==0:
			l=[]
			l.append(name)
			with open("user_files/trusted"+str(us_id)+".csv","a") as f_object:
				writer_object = writer(f_object)
				writer_object.writerow(l)
			ss="Source added"
		elif flag==1:
			ss="Source not added"
		return render_template("ad.html",r=ss)
	return render_template('ad.html')

'''
@app.route('/getT/Res',methods=['GET','POST'])
	def Res():
		if request.method == 'POST':
			print("We're here")

			return render_template("Res.html")
'''
if __name__ == "__main__":
	db.create_all()
	app.run(debug=True)
	

