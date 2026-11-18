#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


data=pd.read_csv(r"D:\spam.csv",encoding="latin-1")
print(data)
data=data.iloc[:,:2]
data.columns = ["label", "message"]
print(data)


# In[3]:


data["label"] = data["label"].map({
    "ham": 0,
    "spam": 1
})

print(data.head())


# In[4]:


x = data["message"]
y = data["label"]

print("X",x)
print("y",y)


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
print(X_train)
print("X_test",X_test)


# In[6]:


tfidf = TfidfVectorizer(stop_words="english")

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("X_train TF-IDF shape:", X_train_tfidf.shape)
print("X_test TF-IDF shape:", X_test_tfidf.shape)


# In[7]:


model = MultinomialNB()


model.fit(X_train_tfidf, y_train)

print("Model training completed successfully!")


# In[8]:



y_pred = model.predict(X_test_tfidf)

print("Predictions completed!")
print(y_pred[:20])


# In[9]:


accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Accuracy percentage:", accuracy * 100, "%")


# In[10]:


print("Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Ham", "Spam"]
))


# In[ ]:


message = input("Enter a message: ")


message_tfidf = tfidf.transform([message])


prediction = model.predict(message_tfidf)

if prediction[0] == 1:
    print("🚨 Prediction: SPAM")
else:
    print("✅ Prediction: HAM")


# In[ ]:


message = "Hi, I will come to college tomorrow."

message_tfidf = tfidf.transform([message])

prediction = model.predict(message_tfidf)

print("Message:", message)

if prediction[0] == 1:
    print("Prediction: SPAM")
else:
    print("Prediction: HAM")


# In[ ]:





# In[ ]:




