#!/usr/bin/env python
# coding: utf-8

# In[52]:


import os
import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVC


# # Data Preprocessing

# In[3]:


spam = pd.read_csv("SMSSpamCollection.csv", header=0, names=['spamorham', 'text'])
print(spam.shape)
spam.head(10)


# # Exploratory Data Analysis

# In[ ]:


# creating a target col, with 0 for ham, 1 for spam
df = spam.drop(columns="spamorham")
df["spam"] = spam["spamorham"].apply(lambda x:1 if 'spam' in str(x)
                                  else 0)
df.columns=['text','target']
df.head(10)


# In[ ]:


df['target'].mean()*100


# In[6]:


#creating two seperate dfs: 1 for spam and 1 for non spam messages only
df_s = df.loc[ df['target']==1]
df_ns = df.loc[ df['target']==0]
    
df_s['len'] = [len(x) for x in df_s["text"]]
spamavg = df_s.len.mean()
print('df_s.head(5)')
print(df_s.head(5))

print('\n\ndf_ns.head(5)')
df_ns['len'] = [len(x) for x in df_ns["text"]]
nonspamavg = df_ns.len.mean()
print(df_ns.head(5))


# In[7]:


spamavg


# In[8]:


nonspamavg


# In[9]:


df['length'] = df['text'].apply(lambda x: len(''.join([a for a in x if a.isdigit()])))

print(np.mean(df['length'][df['target'] == 0]), np.mean(df['length'][df['target'] == 1]))

print(df.head(10))


# # Data Modelling
# MNNB Model Fitting 1

# In[12]:


#train test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], 
                                                    df['target'], 
                                                    random_state=0)


# In[14]:


#fitting and transforming X_train using a Count Vectorizer with default parameters
vect = CountVectorizer().fit(X_train)
X_train_vectorized = vect.transform(X_train)


# In[17]:


vect


# In[19]:


X_train_vectorized


# In[21]:


#fitting a multinomial Naive Bayes Classifier Model with smoothing alpha=0.1
model = sklearn.naive_bayes.MultinomialNB(alpha=0.1)
model_fit = model.fit(X_train_vectorized, y_train)


# In[23]:


#making predictions & looking at AUC score
predictions = model.predict(vect.transform(X_test))
aucscore = roc_auc_score(y_test, predictions) #good!
aucscore


# In[34]:


#confusion matrix
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()
print(pd.DataFrame(confusion_matrix(y_test, predictions),
             columns=['Predicted Spam', "Predicted Ham"], index=['Actual Spam', 'Actual Ham']))

print(f'\nTrue Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')

print(f'\nTrue Positive Rate: { (tp / (tp + fn))}')
print(f'Specificity: { (tn / (tn + fp))}')
print(f'False Positive Rate: { (fp / (fp + tn))}')


# # SVC Model

# In[38]:


#defining an additional function
def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')


# In[40]:


#fit and transfor x_train and X_test
vectorizer = TfidfVectorizer(min_df=5)

X_train_transformed = vectorizer.fit_transform(X_train)
X_train_transformed_with_length = add_feature(X_train_transformed, X_train.str.len())

X_test_transformed = vectorizer.transform(X_test)
X_test_transformed_with_length = add_feature(X_test_transformed, X_test.str.len())
        


# In[46]:


# SVM creation
clf = SVC(C=10000)

clf.fit(X_train_transformed_with_length, y_train)


# In[48]:


y_predicted = clf.predict(X_test_transformed_with_length)


# In[50]:


roc_auc_score(y_test, y_predicted)


# In[53]:


#confusion matrix
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_predicted).ravel()
print(pd.DataFrame(confusion_matrix(y_test, y_predicted),
             columns=['Predicted Spam', "Predicted Ham"], index=['Actual Spam', 'Actual Ham']))
print(f'\nTrue Positives: {tp}')
print(f'False Positives: {fp}')
print(f'True Negatives: {tn}')
print(f'False Negatives: {fn}')


print(f'True Positive Rate: { (tp / (tp + fn))}')
print(f'Specificity: { (tn / (tn + fp))}')
print(f'False Positive Rate: { (fp / (fp + tn))}')


# In[60]:


import seaborn as sb
import matplotlib.pyplot as plt

label = ['MNNB 1', 'SVC',]
auclist = [0.9615532083312719, 0.97422863173865]

#generates an array of length label and use it on the X-axis
def plot_bar_x():
    # this is for plotting purpose
    index = np.arange(len(label))
    clrs = ['grey' if (x < max(auclist)) else 'red' for x in auclist ]
    g=sb.barplot(x=index, y=auclist, palette=clrs) # color=clrs)   
    plt.xlabel('Model type', fontsize=10)
    plt.ylabel('AUC score', fontsize=10)
    plt.xticks(index, label, fontsize=10, rotation=30)
    plt.title('AUC score for each fitted model')
    ax=g
    for p in ax.patches:
             ax.annotate("%.2f" % p.get_height(), (p.get_x() + p.get_width() / 2., p.get_height()),
                 ha='center', va='center', fontsize=11, color='gray', xytext=(0, 20),
                 textcoords='offset points')
    g.set_ylim(0,1.25) #To make space for the annotations

plot_bar_x()


# In[ ]:




