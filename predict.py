import joblib
from preprocessing import clean_text

# Load model & vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_news(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    prediction = model.predict(vec)
    return "FAKE" if prediction[0] == 1 else "REAL"

# Example
if __name__ == "__main__":
    news = "The government has announced a new policy to help students."
    print("Prediction:", predict_news(news))
