#Import important libraries
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
#Load the data
data=pd.read_csv('data/emails.csv')
x=data['text']
y=data['spam']
#Load the TF-IDF vectorizer
vectorizer=TfidfVectorizer(stop_words='english')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
x_train_vect=vectorizer.fit_transform(x_train)
x_test_vect=vectorizer.transform(x_test)
#Load the model
model=SVC(kernel='linear')
model.fit(x_train_vect,y_train)
#Test Accuracy
predictions=model.predict(x_test_vect)
accuracy=accuracy_score(y_test,predictions)
print(f"Accuracy:{accuracy}")
#Prediction
while True:
    custom_email=input("Enter your Email: ")
    custom_email_vect=vectorizer.transform([custom_email])
    prediction=model.predict(custom_email_vect)
    if prediction==1:
        print("Prediction: Spam")
    elif prediction==0:
        print("Prediction: Not spam")
    else:
        print("Error!")