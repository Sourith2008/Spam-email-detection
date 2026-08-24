import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_data(path):
    df=pd.read_csv(path)
    return df

def prepare_data(data):
    x=data["text"]
    y=data["spam"]
    x_train,x_test,y_train,y_test=train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42
    )
    vectorizer=TfidfVectorizer(
        stop_words='english'
    )
    x_train_vect=vectorizer.fit_transform(x_train)
    x_test_vect=vectorizer.transform(x_test)
    return(
        x_train_vect,
        x_test_vect,
        y_train,
        y_test,
        vectorizer
    )