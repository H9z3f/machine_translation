# -*- coding: utf-8 -*-
"""Train Seq2Seq Model for Machine Translation"""
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, RepeatVector
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import optimizers
import pickle

# ==============================
# Configurations
# ==============================
DATA_FILE = '../data/data.txt'
MODEL_FILE = '../models/seq2seq_model.keras'
ENG_TOKENIZER_FILE = '../models/eng_tokenizer.pkl'
DEU_TOKENIZER_FILE = '../models/deu_tokenizer.pkl'

NUM_SAMPLES = 120000  # Number of sentence pairs to use
MAX_LENGTH = 8  # Maximum sentence length (padding)
EMBEDDING_DIM = 512  # Embedding dimension
BATCH_SIZE = 512  # Batch size
EPOCHS = 120  # Number of epochs
VALIDATION_SPLIT = 0.2  # Validation data fraction

# ==============================
# Functions
# ==============================
def read_text(filename):
    """Read text from file."""
    with open(filename, mode='rt', encoding='utf-8') as file:
        return file.read()

def preprocess_data(text):
    """Split and preprocess the text."""
    # Split text into sentence pairs
    lines = text.strip().split('\n')
    pairs = [line.split('\t')[:2] for line in lines if len(line.split('\t')) >= 2]
    # Remove punctuation and lowercase
    for i in range(len(pairs)):
        pairs[i][0] = pairs[i][0].translate(str.maketrans('', '', string.punctuation)).lower()
        pairs[i][1] = pairs[i][1].translate(str.maketrans('', '', string.punctuation)).lower()
    return np.array(pairs)

def create_tokenizer(lines):
    """Create and fit tokenizer on text lines."""
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def encode_and_pad_sequences(tokenizer, lines, max_length):
    """Encode and pad sequences to a uniform length."""
    sequences = tokenizer.texts_to_sequences(lines)
    return pad_sequences(sequences, maxlen=max_length, padding='post')

def build_model(input_vocab, output_vocab, input_length, output_length, embedding_dim):
    """Define Seq2Seq model architecture."""
    model = Sequential([
        Embedding(input_dim=input_vocab, output_dim=embedding_dim, mask_zero=True),
        LSTM(embedding_dim),
        RepeatVector(output_length),
        LSTM(embedding_dim, return_sequences=True),
        Dense(output_vocab, activation='softmax')
    ])
    return model

# ==============================
# Main Program
# ==============================
if __name__ == "__main__":
    # Load and preprocess data
    raw_data = read_text(DATA_FILE)
    sentence_pairs = preprocess_data(raw_data)[:NUM_SAMPLES]
    
    # Tokenize and create vocabulary
    eng_tokenizer = create_tokenizer(sentence_pairs[:, 0])
    deu_tokenizer = create_tokenizer(sentence_pairs[:, 1])
    
    eng_vocab_size = len(eng_tokenizer.word_index) + 1
    deu_vocab_size = len(deu_tokenizer.word_index) + 1
    
    print(f"English Vocabulary Size: {eng_vocab_size}")
    print(f"German Vocabulary Size: {deu_vocab_size}")
    
    # Encode and pad sequences
    train, test = train_test_split(sentence_pairs, test_size=0.2, random_state=42)
    trainX = encode_and_pad_sequences(deu_tokenizer, train[:, 1], MAX_LENGTH)
    trainY = encode_and_pad_sequences(eng_tokenizer, train[:, 0], MAX_LENGTH)
    testX = encode_and_pad_sequences(deu_tokenizer, test[:, 1], MAX_LENGTH)
    testY = encode_and_pad_sequences(eng_tokenizer, test[:, 0], MAX_LENGTH)
    
    # Build and compile model
    model = build_model(deu_vocab_size, eng_vocab_size, MAX_LENGTH, MAX_LENGTH, EMBEDDING_DIM)
    model.compile(optimizer=optimizers.RMSprop(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    
    # Set up checkpoint to save the best model
    checkpoint = ModelCheckpoint(filepath=MODEL_FILE, monitor='val_loss', save_best_only=True, verbose=1)
    
    # Train the model
    history = model.fit(
        trainX,
        trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=VALIDATION_SPLIT,
        callbacks=[checkpoint]
    )
    
    # Plot training and validation loss
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Save tokenizers
    with open(ENG_TOKENIZER_FILE, 'wb') as f:
        pickle.dump(eng_tokenizer, f)
    with open(DEU_TOKENIZER_FILE, 'wb') as f:
        pickle.dump(deu_tokenizer, f)
    
    print("Training complete. Model and tokenizers saved.")
