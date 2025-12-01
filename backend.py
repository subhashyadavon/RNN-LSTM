# ---------------------
# backend.py
# ---------------------

from typing import Any, Dict, Tuple, List
import re
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from collections import Counter
from itertools import chain


# Below are some hyperparameters that you can change or utilize in your functions.

MAX_LEN = 200  # Maximum sequence length (for truncating longer ones)
TRAIN_BATCH_SIZE = 32
TEST_BATCH_SIZE = 64
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

# Notes: 
# - See backend-doc.txt for more information.
# - You can import other modules, define new variables, implement other functions
# - You will also need to define a new class e.g. SentimentClassifier(nn.Module) 
# to represent your model which "build_model" function is supposed to create.


def load_data() -> Tuple[Any, Any]:

    # Load IMDB dataset.
    dataset_dict = load_dataset("imdb")

    # Extract and return train and test splits
    train_split = dataset_dict["train"]
    test_split = dataset_dict["test"]
    return train_split, test_split


def tokenize(example: Dict[str, Any]) -> Dict[str, Any]:

    text = example["text"]

    # Lowercase
    text = text.lower()

    # Remove common HTML artifacts in IMDB reviews
    text = re.sub(r"<br\s*/?>", " ", text)
    text = re.sub(r"</?[^>]+>", " ", text)  # generic HTML tags

    # Optionally normalize abnormal whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Token pattern:
    #   - words with optional internal apostrophe (don't, it's)
    #   - numbers
    #   - punctuation marks as separate tokens
    token_pattern = r"[a-z]+(?:'[a-z]+)?|\d+|[!?.,;:()]"
    tokens: List[str] = re.findall(token_pattern, text)

    return {"tokens": tokens}

# ---------------

def build_vocab(tokenized_train: Any) -> Tuple[Dict[str, int], Dict[str, int]]:

    # Collect all tokens from the training dataset
    all_tokens = list(chain.from_iterable(tokenized_train["tokens"]))
    
    # Count token frequencies
    token_counts = Counter(all_tokens)
    
    # Create vocabulary with special tokens
    special_tokens = {
        "pad": 0,
        "unk": 1
    }
    
    # Build vocab: start with ID 2 for regular tokens
    vocab = {"<PAD>": 0, "<UNK>": 1}
    
    # Add tokens sorted by frequency (most common first)
    for idx, (token, _) in enumerate(token_counts.most_common(), start=2):
        vocab[token] = idx
    
    return vocab, special_tokens


def encode(example: Dict[str, Any], vocab: Dict[str, int], special_tokens: Dict[str, int]) -> Dict[str, Any]:

    tokens = example["tokens"]
    unk_id = special_tokens["unk"]
    pad_id = special_tokens["pad"]
    
    # Convert tokens to IDs, using unk_id for unknown tokens
    input_ids = [vocab.get(token, unk_id) for token in tokens]
    
    # Truncate if longer than MAX_LEN
    if len(input_ids) > MAX_LEN:
        input_ids = input_ids[:MAX_LEN]
    
    # Pad if shorter than MAX_LEN
    while len(input_ids) < MAX_LEN:
        input_ids.append(pad_id)
    
    return {"input_ids": input_ids}


def prepare_datasets(raw_datasets: Tuple[Any, Any], vocab: Dict[str, int], special_tokens: Dict[str, int]) -> Tuple[Any, Any]:

    train_raw, test_raw = raw_datasets
    
    # Tokenize the datasets
    train_tokenized = train_raw.map(tokenize)
    test_tokenized = test_raw.map(tokenize)
    
    # Encode the datasets (convert tokens to IDs)
    encode_fn = lambda example: encode(example, vocab, special_tokens)
    train_encoded = train_tokenized.map(encode_fn)
    test_encoded = test_tokenized.map(encode_fn)
    
    # Format for PyTorch: convert to tensors
    train_encoded.set_format(type="torch", columns=["input_ids", "label"])
    test_encoded.set_format(type="torch", columns=["input_ids", "label"])
    
    return train_encoded, test_encoded



# Define LSTM-based sentiment classifier model
class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.5):
        super(SentimentClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected output layer (bidirectional LSTM outputs hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, 2)  # 2 classes: positive/negative
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, input_ids):
        # input_ids shape: [batch_size, seq_len]
        
        # Embedding
        embedded = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)  # lstm_out: [batch_size, seq_len, hidden_dim*2]
        
        # Use the last hidden state from both directions
        # hidden shape: [num_layers*2, batch_size, hidden_dim]
        # Concatenate the final forward and backward hidden states
        hidden_fwd = hidden[-2, :, :]  # Forward direction last layer
        hidden_bwd = hidden[-1, :, :]  # Backward direction last layer
        hidden_concat = torch.cat([hidden_fwd, hidden_bwd], dim=1)  # [batch_size, hidden_dim*2]
        
        # Apply dropout
        hidden_concat = self.dropout(hidden_concat)
        
        # Fully connected layer
        output = self.fc(hidden_concat)  # [batch_size, 2]
        
        return output


def build_model(vocab_size: int) -> Any:

    # Create the model
    model = SentimentClassifier(vocab_size)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    return model


def train_model(model: Any, train_dataset: Any) -> Any:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
    
    return model


def evaluate(model: Any, test_dataset: Any) -> float:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            labels = batch["label"].to(device)
            
            # Forward pass
            outputs = model(input_ids)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Update counts
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


def predict_sentiment(model: Any, text: str, vocab: Dict[str, int], special_tokens: Dict[str, int]) -> str:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Tokenize the input text
    example = {"text": text}
    tokenized = tokenize(example)
    
    # Encode the tokens
    encoded = encode(tokenized, vocab, special_tokens)
    
    # Convert to tensor and add batch dimension
    input_ids = torch.tensor(encoded["input_ids"]).unsqueeze(0).to(device)  # [1, MAX_LEN]
    
    # Get prediction
    with torch.no_grad():
        output = model(input_ids)
        _, predicted = torch.max(output, 1)
    
    # Convert prediction to sentiment string
    sentiment = "positive" if predicted.item() == 1 else "negative"
    
    return sentiment

