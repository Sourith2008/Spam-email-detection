from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def get_models():
    models={
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        )
    }
    return models