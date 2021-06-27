import pandas as pd
'''
t=input()
y=t.strip("!.,")

x=i.split()
print(y)
print(x)'''
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from csv import writer
# X = input("Enter first string: ").lower()
# Y = input("Enter second string: ").lower()
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
  
# cosine formula 
    for i in range(len(rvector)):
            c+= l1[i]*l2[i]
    cosine = c / float((sum(l1)*sum(l2))**0.5)
    return cosine

X ="Bangladesh is not beating india in everything except cricket"
Y ="Bangladesh is beating india in everything except cricket"
o=sim(X,Y)
pe=pd.read_csv("Datasets/nee.csv")
flag=0
for i in list(pe.Headlines):
    if flag==0:
        u=sim(X,i)
    if u>=0.8:
        flag=1
if flag==0:
    l=[]
    l.append(X)
    with open('Datasets/nee.csv', 'a') as f_object:
        writer_object = writer(f_object)
        writer_object.writerow(l)
print(o)
# tokenization
