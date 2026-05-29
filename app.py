from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# =========================
# CLEAN THE DATASET
# =========================

with open("data.csv", "r", encoding="latin-1") as f:
    lines = f.readlines()

clean_data = []

for line in lines:
    line = line.strip()
    line = line.replace('"', '')

    parts = line.split(maxsplit=1)

    if len(parts) == 2:
        label = parts[0].lower()
        message = parts[1]

        if label in ["ham", "spam"]:
            clean_data.append([label, message])

df = pd.DataFrame(clean_data, columns=["label", "message"])

# =========================
# TRAIN MODEL
# =========================

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df['message'])

y = df['label']

model = MultinomialNB()
model.fit(X, y)

# =========================
# FRONTEND
# =========================

@app.route("/", methods=["GET", "POST"])
def home():

    result = ""

    if request.method == "POST":

        user_msg = request.form["message"]

        msg_vec = vectorizer.transform([user_msg])

        prediction = model.predict(msg_vec)

        if prediction[0] == "spam":
            result = "Spam Message"
        else:
            result = "Not Spam Message"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)

