from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample dataset
texts = [
    "Win money now", "Buy cheap pills", "Hello friend", "Meeting at 10 am",
    "Win a free vacation", "Call mom", "Cheap loans available", "Let's catch up soon",
    "Earn money quickly", "Important project deadline"
]
labels = [1, 1, 0, 0, 1, 0, 1, 0, 1, 0]  # 1: spam, 0: not spam

# Step 1: Convert text to numerical vectors
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Step 3: Train Naive Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 4: Predict
y_pred = model.predict(X_test)

# Step 5: Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
