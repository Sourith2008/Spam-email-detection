from preprocessing import load_data, prepare_data
from models import get_models
from evaluation import evaluate_model,show_confusion_matrix
import joblib

ORIGINAL_DATA='data/emails.csv'
EXPANSION_DATA='data/modern_email_expansion.csv'

data=load_data(ORIGINAL_DATA,EXPANSION_DATA)

x_train,x_test,y_train,y_test,vectorizer=prepare_data(data)

models=get_models()

results=[]
for model_name, model in models.items():
    print(f'Training {model_name}')

    model.fit(x_train,y_train)
    metrics=evaluate_model(
        model,
        x_test,
        y_test,
        model_name
    )
    results.append({
        "Model": model_name,
        **metrics
    })
    show_confusion_matrix(model,x_test,y_test,model_name)

print("MODEL COMPARISION")

for result in results:
        print(
            f"{result["Model"]:<20}"
            f"Accuracy: {result['Accuracy']:.4f}"
            f"Precision: {result['Precision']:.4f}"
            f"Recall: {result['Recall']:.4f}"
            f"F1-score: {result['F1 Score']:.4f}"
        )

best_model=models["SVM"]
joblib.dump(best_model,'models/best_model.pkl')
joblib.dump(vectorizer,'models/tfidf_vectorizer.pkl')