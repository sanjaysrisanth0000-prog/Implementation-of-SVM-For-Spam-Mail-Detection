# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Step 1:
Import required libraries and load the dataset spam.csv using pandas. Select the columns label and message, and convert labels ham → 0 and spam → 1.

Step 2:
Convert the text messages into numerical features using TF-IDF Vectorization with scikit-learn.

Step 3:
Split the dataset into training and testing sets, then train the model using the Support Vector Machine with a linear kernel.

Step 4:
Predict the test data, compute the confusion matrix, and visualize the results using Seaborn and Matplotlib.

## Program:
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SANJAY SRISANTH V
RegisterNumber:  25018855
*/
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1','v2']]
data.columns = ['label','message']

# Convert labels
data['label'] = data['label'].map({'ham':0, 'spam':1})

# Features and target
X = data['message']
y = data['label']

# TF-IDF conversion
vectorizer = TfidfVectorizer(stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, random_state=42)

# Train SVM
model = SVC(kernel='linear')
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Ham','Spam'],
            yticklabels=['Ham','Spam'])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix for SVM Spam Detection")
plt.show()
```

## Output:
![SVM For Spam Mail Detection](sam.png)
cluster:
0.    2
1.    3
2.    2
3.    3
4.    2
## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
