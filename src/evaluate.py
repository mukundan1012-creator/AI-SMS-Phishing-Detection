import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay
)


def evaluate_model(model, X_test, y_test, save_plot_path=None):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(classification_report(y_test, predictions))

    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, predictions,
        display_labels=["Ham", "Spam"],
        cmap="Blues"
    )
    plt.title("Confusion Matrix — Logistic Regression")

    if save_plot_path:
        plt.savefig(save_plot_path, dpi=150, bbox_inches="tight")
    plt.show()

    return accuracy, roc_auc