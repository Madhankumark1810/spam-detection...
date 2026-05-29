import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# =========================
# CLEAN THE DATASET
# =========================

# Read original dataset
with open("data.csv", "r", encoding="latin-1") as f:
    lines = f.readlines()

clean_data = []

for line in lines:
    line = line.strip()

    # Remove quotes
    line = line.replace('"', '')

    # Split first word (ham/spam) and message
    parts = line.split(maxsplit=1)

    # Check valid rows
    if len(parts) == 2:
        label = parts[0].lower()
        message = parts[1]

        # Keep only ham/spam rows
        if label in ["ham", "spam"]:
            clean_data.append([label, message])

# Create dataframe
df = pd.DataFrame(clean_data, columns=["label", "message"])

# Save cleaned dataset
df.to_csv("cleaned_data.csv", index=False)

print("Dataset cleaned successfully!")

# =========================
# TRAIN MODEL
# =========================

# Convert text into numbers
vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['message'])

# Labels
y = df['label']

# Train Naive Bayes model
model = MultinomialNB()
model.fit(X, y)

print("Model trained successfully!")

# =========================
# USER INPUT
# =========================

user_msg = input("\nEnter your message: ")

# Convert user message
msg_vec = vectorizer.transform([user_msg])

# Predict
prediction = model.predict(msg_vec)

# Output result
print("\nPrediction Result:")

if prediction[0] == "spam":
    print("Spam Message")
else:
    print("Not Spam Message")