import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


texts = [
    "I love this product",
    "This is the best experience",
    "I hate it",
    "This is terrible",
    "Absolutely fantastic service",
    "Not good at all"
]
labels = [1, 1, 0, 0, 1, 0]

# Parameters
max_words = 1000
max_len = 10

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

data = pad_sequences(sequences, maxlen=max_len)

labels = np.array(labels)

model = Sequential([
    Embedding(input_dim=max_words, output_dim=32),
    SimpleRNN(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

print("Training the model...")
model.fit(data, labels, epochs=10, batch_size=2)

test_texts = ["I absolutely love this", "This is the worst thing ever"]
test_sequences = tokenizer.texts_to_sequences(test_texts)
test_data = pad_sequences(test_sequences, maxlen=max_len)

predictions = model.predict(test_data)
for text, pred in zip(test_texts, predictions):
    sentiment = "Positive" if pred > 0.5 else "Negative"
    print(f"Text: \"{text}\" - Sentiment: {sentiment}")
