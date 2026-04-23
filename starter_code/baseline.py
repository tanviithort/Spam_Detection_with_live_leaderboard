# baseline.py
# Simple Spam Detection Baseline Model

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --------------------------------------------------
# Sample training data
# Replace with a larger dataset later if needed
# --------------------------------------------------

data = {
    "message": [
        "Congratulations you won a free iPhone",
        "Claim your lottery prize now",
        "Free entry in a cash contest",
        "Call me when you reach home",
        "Let's meet tomorrow for lunch",
        "Can you send the assignment",
        "Win money now click this link",
        "Urgent your account has won reward",
        "Are we still meeting today",
        "Please bring the documents"
    ],
    "label": [
        "spam",
        "spam",
        "spam",
        "ham",
        "ham",
        "ham",
        "spam",
        "spam",
        "ham",
        "ham"
    ]
}

df = pd.DataFrame(data)

# --------------------------------------------------
# Train/Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.3,
    random_state=42
)

# --------------------------------------------------
# Baseline Pipeline
# CountVectorizer + Naive Bayes
# --------------------------------------------------

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB())
])

# Train
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)

# Evaluate
print("\nAccuracy:", accuracy_score(y_test, preds))
print("\nClassification Report:\n")
print(classification_report(y_test, preds))


# --------------------------------------------------
# Predict on challenge test messages
# Replace with your actual challenge test set
# --------------------------------------------------

test_messages = [
    "You won a free vacation claim now",
    "Can we meet at 5 pm",
    "Limited offer click here",
    "Please send notes"
]

predictions = model.predict(test_messages)

submission = pd.DataFrame({
    "id": range(1, len(test_messages)+1),
    "label": predictions
})

submission.to_csv(
    "submissions/baseline_submission.csv",
    index=False
)

print("\nSubmission file generated:")
print("submissions/baseline_submission.csv")
print(submission)
