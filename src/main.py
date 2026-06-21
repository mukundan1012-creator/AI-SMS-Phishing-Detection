from train_model import train_and_save
from evaluate import evaluate_model

model, tfidf, X_test, y_test = train_and_save()

print("=== Logistic Regression (Final Model) ===")
evaluate_model(model, X_test, y_test, save_plot_path="confusion_matrix.png")
