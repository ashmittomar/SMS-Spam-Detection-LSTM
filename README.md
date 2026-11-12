# SMS Spam and Ham Detection Using LSTM and NLP

## Project Overview

This project implements an SMS Spam and Ham Detection system using Natural Language Processing (NLP) and a Long Short-Term Memory (LSTM) neural network.

The system analyzes SMS messages and classifies them as **Spam** or **Ham (Genuine)**. It uses text preprocessing, tokenization, sequence padding and deep learning to identify suspicious message patterns.

## Objectives

- Analyze and preprocess SMS text data.
- Apply NLP techniques for text cleaning and tokenization.
- Develop an LSTM-based binary classification model.
- Evaluate model performance using standard classification metrics.
- Classify new SMS messages through a real-time prediction interface.

## Technologies Used

- **Programming Language:** Python
- **Deep Learning:** TensorFlow, Keras, LSTM
- **NLP:** NLTK, Regular Expressions
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook, VS Code

## Dataset

The project uses the SMS Spam Collection dataset from the UCI Machine Learning Repository.

- Original dataset: 5,572 SMS records
- Records after duplicate removal: 5,169
- Ham messages: 4,516
- Spam messages: 653

## Project Workflow

1. Dataset exploration and analysis
2. Data cleaning and duplicate removal
3. Text preprocessing and normalization
4. Label encoding (Ham = 0, Spam = 1)
5. Tokenization and sequence padding
6. Training, validation and testing split
7. LSTM model development
8. Model training with class weighting and early stopping
9. Model evaluation
10. Real-time SMS classification

## Model Architecture

The LSTM model consists of:

- Embedding Layer (64 dimensions)
- LSTM Layer (64 units)
- Dropout Layer (0.3)
- Dense Layer (1 neuron, Sigmoid activation)

## Dataset Split

- Training: 72%
- Validation: 8%
- Testing: 20%

## Model Performance

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 97.58% |
| Precision | 90.15% |
| Recall | 90.84% |
| F1-Score | 90.49% |

The model was evaluated on 1,034 unseen test messages.

## Project Structure

```text
SMS-Spam-Detection-LSTM/
├── dataset/
├── models/
├── notebooks/
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation and Setup

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/SMS-Spam-Detection-LSTM.git
```

2. Navigate to the project directory:

```bash
cd SMS-Spam-Detection-LSTM
```

3. Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

4. Install dependencies:

```bash
pip install -r requirements.txt
```

5. Run the application:

```bash
python main.py
```

## Real-Time SMS Detection

Users can enter SMS messages through the terminal interface. The trained LSTM model predicts whether a message is Spam or Ham and displays the prediction confidence and spam probability.

## Future Enhancements

- Develop a Flask-based web application.
- Implement Google authentication.
- Add user-specific SMS analysis history.
- Build an interactive dashboard for message analysis.

## Author

**Ashmit Tomar**

Computer Science Undergraduate | AI & Machine Learning

GitHub: https://github.com/ashmittomar

---

*This project was developed as an academic machine learning project to explore NLP and LSTM-based text classification.*