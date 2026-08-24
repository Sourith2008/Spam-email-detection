from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model,x_test,y_test,model_name):
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    print(f"\nClassifiaction Report of {model_name}")
    print(
        classification_report(
            y_test,y_pred
        )
    )
    metrics={
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }
    return metrics
def show_confusion_matrix(model,x_test,y_test,model_name):
        y_pred=model.predict(x_test)
        cm=confusion_matrix(y_test,y_pred)
        sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=["Not Spam", "Spam"],
        yticklabels=["Not Spam", "Spam"]
        )
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix for {model_name}")
        plt.show()