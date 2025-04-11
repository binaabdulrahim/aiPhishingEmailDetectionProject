import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Load the dataset
csv_path = "../data/emails.csv" # Change this if needed
df = pd.read_csv(csv_path)

# Step 2: Extract features and labels
X = df['text']
y = df['label_num'] # 0 for ham, 1 for spam

# Step 3: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 5: Train a Logistic Regression classifier
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Step 6: Evaluate the model
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Logistic Regression Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)

# Step 7: Save the model and vectorizer
joblib.dump(model, "logistic_regression_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
os.makedirs("../results", exist_ok=True)
with open("../results/lr_model_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy:.2f}\n")
    f.write("Classification Report:\n")
    f.write(report)

# Step 8: Visualize confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ham", "Spam"])
disp.plot(cmap="Blues")
plt.title("Logistic Regression - Confusion Matrix")
plt.savefig("../results/lr_confusion_matrix.png")
plt.close()

# Step 9: Visualize Heart map
report_dict = classification_report(y_test, y_pred, output_dict=True)
sns.heatmap(pd.DataFrame(report_dict).iloc[:-1, :].T, annot=True, cmap="YlGnBu")
plt.title("Logistic Regression - Classification Report Heatmap")
plt.savefig("../results/lr_report_heatmap.png")
plt.close()

# Step 10: Visualize ROC curve
y_prob = model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression - ROC Curve")
plt.legend(loc="lower right")
plt.savefig("../results/lr_roc_curve.png")
plt.close()

print("Visualizations saved to ../results/")