
import os
import re
import pickle
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from tensorflow.keras.preprocessing.sequence import pad_sequences


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "sms_spam_lstm.keras")
DATA_PATH = os.path.join(MODEL_DIR, "preprocessed_data.npz")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")


# ==========================================
# 2. LOAD TRAINED MODEL AND DATA
# ==========================================

print("\nLoading SMS Spam Detection System...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

data = np.load(DATA_PATH)

X_test = data["X_test"]
y_test = data["y_test"].flatten()

with open(TOKENIZER_PATH, "rb") as file:
    tokenizer = pickle.load(file)

MAX_LENGTH = model.input_shape[1]

print("\nAll files loaded successfully!")
print("Model: LSTM")
print("Test Samples:", len(X_test))
print("Vocabulary Size:", len(tokenizer.word_index) + 1)
print("Maximum Sequence Length:", MAX_LENGTH)


# ==========================================
# 3. MODEL EVALUATION
# ==========================================

y_prob = model.predict(X_test, verbose=0).flatten()

y_pred = (y_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)


print("\n" + "=" * 50)
print("          LSTM MODEL EVALUATION")
print("=" * 50)

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1-Score  : {f1 * 100:.2f}%")

print("\n" + "-" * 50)
print("CLASSIFICATION REPORT")
print("-" * 50)

print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["Ham", "Spam"],
        zero_division=0
    )
)


# ==========================================
# 4. CONFUSION MATRIX
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=[0, 1]
)

print("\n" + "-" * 50)
print("CONFUSION MATRIX")
print("-" * 50)

print("                 Predicted")
print("              Ham       Spam")
print(f"Actual Ham    {cm[0][0]:<10}{cm[0][1]}")
print(f"Actual Spam   {cm[1][0]:<10}{cm[1][1]}")


# ==========================================
# 5. TEXT PREPROCESSING
# ==========================================

def preprocess_text(text):

    text = text.lower()

    text = re.sub(
        r'https?://\S+|www\.\S+',
        ' URL ',
        text
    )

    text = re.sub(
        r'\S+@\S+',
        ' EMAIL ',
        text
    )

    text = re.sub(
        r'[^a-z0-9\s]',
        ' ',
        text
    )

    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text


# ==========================================
# 6. SMS PREDICTION FUNCTION
# ==========================================

def predict_sms(message):

    cleaned_message = preprocess_text(message)

    sequence = tokenizer.texts_to_sequences(
        [cleaned_message]
    )

    padded_message = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding='post',
        truncating='post'
    )

    spam_probability = float(
        model.predict(
            padded_message,
            verbose=0
        )[0][0]
    )

    if spam_probability >= 0.5:

        prediction = "SPAM"
        confidence = spam_probability * 100

    else:

        prediction = "HAM"
        confidence = (1 - spam_probability) * 100

    print("\n" + "=" * 50)
    print("          SMS CLASSIFICATION RESULT")
    print("=" * 50)

    print("\nMessage:", message)

    if prediction == "SPAM":
        print("\nRESULT: SPAM / SUSPICIOUS")
    else:
        print("\nRESULT: HAM / GENUINE")

    print(f"Confidence: {confidence:.2f}%")
    print(f"Spam Probability: {spam_probability * 100:.2f}%")

    print("=" * 50)


# ==========================================
# 7. REAL-TIME SMS TESTING
# ==========================================

print("\n" + "=" * 50)
print("          SMS SPAM DETECTION SYSTEM")
print("=" * 50)

print("\nEnter an SMS message to check.")
print("Type 'exit' to close the application.")

while True:

    user_message = input("\nEnter SMS: ").strip()

    if user_message.lower() == "exit":
        print("\nThank you for using SMS Spam Detection!")
        break

    if not user_message:
        print("Please enter a valid SMS message.")
        continue

    predict_sms(user_message)
