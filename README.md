# IMDB Movie Review Sentiment Analysis using SimpleRNN

🚀 **Live Demo:** [Try the IMDB Sentiment Analysis App](https://imdb-movie-review-sentiment-analysis-fm6rdwrboxxdjn8bznuppk.streamlit.app/)

A Deep Learning project that uses a **Simple Recurrent Neural Network (SimpleRNN)** to classify movie reviews as Positive or Negative. The model is trained on the IMDB Movie Reviews dataset using TensorFlow/Keras and deployed as an interactive web application using Streamlit.

## 1. Project Overview

Sentiment analysis is a Natural Language Processing (NLP) task that identifies the sentiment or opinion expressed in textual data.

In this project, a user enters a movie review, and the trained SimpleRNN model predicts whether the review expresses a positive or negative sentiment.

The application follows this pipeline:

1. Accept a movie review from the user.
2. Convert text into lowercase tokens.
3. Convert words into numerical representations using the IMDB word index.
4. Pad or truncate sequences to a fixed length.
5. Pass the processed sequence through the trained neural network.
6. Generate a sentiment score using the sigmoid activation function.
7. Classify the review as Positive or Negative using a threshold of 0.5.

## 2. Technologies and Environment

| Component | Version / Technology |
|---|---|
| Programming Language | Python 3.12 |
| Deep Learning Framework | TensorFlow 2.21.0 |
| Neural Network API | Keras |
| Numerical Computing | NumPy |
| Web Application | Streamlit |
| Dataset | IMDB Movie Reviews |

## 3. Dataset Description

The project uses the **IMDB Movie Reviews dataset**, available through the built-in `tensorflow.keras.datasets.imdb` module.

The dataset contains movie reviews labelled according to their sentiment.

| Dataset Property | Description |
|---|---|
| Total Reviews | 50,000 |
| Training Samples | 25,000 |
| Testing Samples | 25,000 |
| Positive Reviews | 25,000 |
| Negative Reviews | 25,000 |
| Classification Type | Binary Classification |
| Vocabulary Size Used | 10,000 |

The dataset is already divided into training and testing sets, with balanced positive and negative classes.

### Loading the Dataset

```python
from tensorflow.keras.datasets import imdb

max_features = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(
    num_words=max_features
)
```

The reviews are provided as sequences of integers rather than raw text. Each integer represents a word in the dataset vocabulary.

## 4. Data Preprocessing

Before training, the reviews are preprocessed to make them suitable for the neural network.

**Vocabulary Limitation**

Only the 10,000 most frequent words are considered using `num_words=10000`.

**Sequence Padding and Truncation**

Reviews have different lengths, but the neural network requires inputs of a consistent shape.

Therefore, `pad_sequences()` is used to make every review exactly 500 tokens long.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

max_len = 500

X_train = pad_sequences(
    X_train,
    maxlen=max_len,
    padding='pre',
    truncating='pre'
)

X_test = pad_sequences(
    X_test,
    maxlen=max_len,
    padding='pre',
    truncating='pre'
)
```

- `maxlen=500`: Maximum sequence length.
- `padding='pre'`: Adds zeros at the beginning of shorter sequences.
- `truncating='pre'`: Removes tokens from the beginning of longer sequences.

## 5. Model Architecture

The project uses a Simple Recurrent Neural Network (SimpleRNN) for sequential text processing.

### Architecture

| Layer | Configuration | Purpose |
|---|---|---|
| Input | 500 tokens | Accepts preprocessed reviews |
| Embedding | 10,000 vocabulary size, 128 dimensions | Converts word IDs into dense vectors |
| SimpleRNN | 128 units, tanh activation | Learns sequential patterns |
| Dense | 1 neuron, sigmoid activation | Produces sentiment score |

### Model Implementation

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, Dense

model = Sequential([
    Input(shape=(max_len,)),

    Embedding(
        input_dim=max_features,
        output_dim=128,
        mask_zero=True
    ),

    SimpleRNN(
        128,
        activation='tanh'
    ),

    Dense(
        1,
        activation='sigmoid'
    )
])
```

### How the Model Works

**Embedding Layer:** Converts each integer word ID into a trainable 128-dimensional vector representation.

**SimpleRNN Layer:** Processes the sequence of word embeddings and maintains a hidden state that carries information from previous time steps.

**Dense Layer:** Produces a single output using sigmoid activation, giving a value between 0 and 1.

A threshold of 0.5 is used for classification:

- Score > 0.5 → Positive
- Score ≤ 0.5 → Negative

## 6. Training Configuration and Hyperparameters

The following hyperparameters were used during model training:

| Hyperparameter | Value |
|---|---|
| Vocabulary Size (`max_features`) | 10,000 |
| Maximum Sequence Length (`max_len`) | 500 |
| Embedding Dimension | 128 |
| SimpleRNN Units | 128 |
| RNN Activation | tanh |
| Output Activation | sigmoid |
| Optimizer | Adam |
| Learning Rate | 0.0001 |
| Loss Function | Binary Crossentropy |
| Batch Size | 32 |
| Maximum Epochs | 10 |
| Validation Split | 20% |
| Early Stopping Patience | 2 |
| Restore Best Weights | True |

### Model Compilation

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0001
    ),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Early Stopping

Early stopping is used to monitor validation loss during training.

Training stops when validation loss does not improve for two consecutive epochs. The weights from the best validation-loss epoch are restored.

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=2,
    restore_best_weights=True
)
```

### Model Training

```python
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)
```

## 7. Model Evaluation

The trained model is evaluated on the separate test dataset containing 25,000 reviews.

**Test Accuracy: 85.34%**

| Evaluation Metric | Result |
|---|---:|
| Test Accuracy | 85.34% |
| Test Loss | 0.3615 |

The results indicate that the SimpleRNN model can learn useful sequential patterns for binary sentiment classification.

## 8. Streamlit Web Application

The trained model is saved in the `.keras` format and loaded into the Streamlit application for inference.

The application allows users to:

- Enter custom movie reviews.
- Classify reviews as Positive or Negative.
- View the prediction score.

### Run Locally

Clone the repository:

```bash
git clone https://github.com/Manshu418/IMDB-Movie-Review-Sentiment-Analysis.git
```

Navigate to the project directory:

```bash
cd IMDB-Movie-Review-Sentiment-Analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run main.py
```

The application will be available at the local URL provided by Streamlit, usually `http://localhost:8501`.

## 9. Project Structure

```text
IMDB-Movie-Review-Sentiment-Analysis/
│
├── main.py
├── simple_rnn_imdb.keras
├── requirements.txt
├── .gitignore
└── README.md
```

## 10. Future Improvements

- Experiment with LSTM and GRU architectures.
- Compare SimpleRNN, LSTM, and GRU performance.
- Improve text preprocessing and tokenization.
- Experiment with bidirectional recurrent networks.
- Explore Transformer-based models for sentiment analysis.

## 11. Live Application

Try the deployed application:

**[IMDB Movie Review Sentiment Analysis – Live Demo](https://imdb-movie-review-sentiment-analysis-fm6rdwrboxxdjn8bznuppk.streamlit.app/)**

---

**Developed using Python, TensorFlow/Keras, SimpleRNN, and Streamlit.**