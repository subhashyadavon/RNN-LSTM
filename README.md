# LSTM Sentiment Analysis on IMDB Dataset

A PyTorch implementation of a bidirectional LSTM model for sentiment classification on the IMDB movie review dataset.

## 📊 Results

- **Test Accuracy**: 83.14%
- **Model**: Bidirectional LSTM with 2 layers
- **Dataset**: IMDB movie reviews (25,000 train / 25,000 test)

## 🏗️ Model Architecture

```
SentimentClassifier(
  - Embedding Layer: 128-dimensional embeddings
  - Bidirectional LSTM: 2 layers, 256 hidden units per direction
  - Dropout: 0.5 for regularization
  - Fully Connected: Output layer mapping to 2 classes
)
```

### Hyperparameters

- **Max Sequence Length**: 200 tokens
- **Batch Size**: 32 (training) / 64 (testing)
- **Learning Rate**: 1e-3
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss
- **Epochs**: 5

## 📁 Project Structure

```
.
├── backend.py          # Core implementation (model, training, evaluation)
├── frontend.ipynb      # Jupyter notebook with experiments and results
├── frontend.html       # HTML export of executed notebook
└── README.md          # This file
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch datasets
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/subhashyadavon/RNN-LSTM.git
cd RNN-LSTM
```

2. Install dependencies:
```bash
pip install torch datasets
```

### Usage

#### Training and Evaluation

Run the Jupyter notebook:
```bash
jupyter notebook frontend.ipynb
```

Or use the backend directly in Python:

```python
from backend import (
    load_data, build_vocab, prepare_datasets,
    build_model, train_model, evaluate, predict_sentiment
)

# Load and prepare data
train_raw, test_raw = load_data()
train_tokenized = train_raw.map(tokenize)
vocab, special_tokens = build_vocab(train_tokenized)

# Prepare datasets
train_dataset, test_dataset = prepare_datasets(
    (train_raw, test_raw), vocab, special_tokens
)

# Build and train model
model = build_model(vocab_size=len(vocab))
model = train_model(model, train_dataset)

# Evaluate
accuracy = evaluate(model, test_dataset)
print(f"Test Accuracy: {accuracy:.4f}")
```

#### Making Predictions

```python
# Predict sentiment for custom text
text = "This movie was absolutely wonderful!"
sentiment = predict_sentiment(model, text, vocab, special_tokens)
print(f"Sentiment: {sentiment}")  # Output: positive
```

## 📈 Training Progress

| Epoch | Loss   |
|-------|--------|
| 1/5   | 0.6742 |
| 2/5   | 0.5719 |
| 3/5   | 0.3542 |
| 4/5   | 0.2445 |
| 5/5   | 0.1602 |

## 🎯 Example Predictions

| Review Text | Predicted Sentiment |
|-------------|-------------------|
| "This movie was absolutely wonderful, I loved every minute of it." | ✅ Positive |
| "The film was boring and a complete waste of time." | ❌ Negative |
| "Good movie." | ✅ Positive |
| "It was terrible." | ❌ Negative |
| "Acting was breathtaking." | ✅ Positive |

## 🔧 Implementation Details

### Backend Functions

- **`load_data()`**: Loads IMDB dataset from HuggingFace
- **`tokenize()`**: Tokenizes text with HTML tag removal and normalization
- **`build_vocab()`**: Builds frequency-based vocabulary with special tokens
- **`encode()`**: Converts tokens to fixed-length ID sequences
- **`prepare_datasets()`**: Prepares PyTorch-ready datasets
- **`build_model()`**: Constructs the LSTM model
- **`train_model()`**: Trains the model with backpropagation
- **`evaluate()`**: Computes test accuracy
- **`predict_sentiment()`**: Predicts sentiment for raw text

### Key Features

- ✅ Bidirectional LSTM for better context understanding
- ✅ Automatic GPU/CPU detection and usage
- ✅ Dropout regularization to prevent overfitting
- ✅ Frequency-based vocabulary building
- ✅ Proper padding and truncation handling
- ✅ HTML tag cleaning for IMDB reviews

## 📝 Dataset

The IMDB dataset contains 50,000 movie reviews:
- **Training Set**: 25,000 reviews
- **Test Set**: 25,000 reviews
- **Classes**: Binary (Positive/Negative)
- **Vocabulary Size**: 80,343 unique tokens

## 🛠️ Technologies Used

- **PyTorch**: Deep learning framework
- **HuggingFace Datasets**: Dataset loading and processing
- **Python 3.9+**: Programming language
- **Jupyter**: Interactive development environment

## 📄 License

This project is for educational purposes as part of an assignment.

## 👤 Author

Subhash Yadav

## 🙏 Acknowledgments

- IMDB dataset from HuggingFace
- PyTorch documentation and tutorials
