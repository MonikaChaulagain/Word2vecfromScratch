# Word2Vec Skip-gram Implementation from Scratch

A complete implementation of Word2Vec using the skip-gram architecture with negative sampling, built entirely from scratch using Python and NumPy without any external machine learning frameworks.

## 🚀 Features

- **Skip-gram Architecture**: Predicts context words given a target word
- **Negative Sampling**: Efficient training technique that samples negative examples instead of computing full softmax
- **Custom Vocabulary Building**: Handles word frequency filtering and vocabulary management
- **Unigram Table**: Optimized negative sampling based on word frequency distribution
- **Cosine Similarity Search**: Find most similar words using trained embeddings
- **Model Persistence**: Save and load trained models
- **Educational Implementation**: Clear, well-documented code for learning purposes

## 📋 Requirements

```bash
pip install numpy matplotlib tqdm
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/MonikaChaulagain/Word2vecfromScratch.git
cd Word2vecfromScratch
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the example:
```bash
python word2vec_skipgram.py
```

## 📖 Usage

### Basic Usage

```python
from word2vec_skipgram import Word2Vec

# Initialize model
model = Word2Vec(
    vector_size=100,
    window_size=5,
    negative_samples=5,
    learning_rate=0.025,
    min_count=5,
    epochs=10
)

# Prepare your corpus (list of sentences)
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "park"],
    # ... more sentences
]

# Train the model
model.train(sentences)

# Find similar words
similar_words = model.most_similar("cat", topn=5)
print(similar_words)

# Get word vector
vector = model.get_word_vector("cat")

# Save model
model.save_model("my_word2vec.pkl")
```

### Custom Corpus

Replace the `create_sample_corpus()` function with your own data:

```python
def load_custom_corpus():
    with open('your_text_file.txt', 'r', encoding='utf-8') as f:
        corpus = f.readlines()
    return corpus
```

## ⚙️ Parameters

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `vector_size` | Dimension of word embeddings | 100 | 50-300 |
| `window_size` | Context window size | 5 | 3-10 |
| `negative_samples` | Number of negative samples | 5 | 5-20 |
| `learning_rate` | Initial learning rate | 0.025 | 0.01-0.1 |
| `min_count` | Minimum word frequency | 5 | 1-10 |
| `epochs` | Number of training epochs | 10 | 5-20 |

## 📊 Example Output

```
Word2Vec skip gram implementation
========================================
Creating sample corpus...
Preprocessing corpus...
Starting Training....
Building vocabulary
Vocabulary size: 34
Building unigram table for negative sampling
Unigram table size: 234287473
Initializing weights...
Generated 25200 training pairs

Epoch 1/10: 100%|██████████| 25200/25200 [00:02<00:00, 9189.89it/s]
Learning rate: 0.023750
...

========================================
Testing the model:
========================================

Most similar to 'monika':
  filthy: 0.8555
  amazing: 0.5639
  chaulagain: 0.4551
  is: 0.3355

Most similar to 'friend':
  great: 0.4753
  person: 0.3152
  has: 0.1464
```

## 🧠 Algorithm Overview

### Skip-gram Model
The skip-gram model learns word representations by predicting context words given a target word. For each word in the corpus, it tries to predict the surrounding words within a specified window.

### Negative Sampling
Instead of computing the expensive softmax over the entire vocabulary, negative sampling randomly selects a few "negative" words that shouldn't appear in the context and trains the model to distinguish between positive and negative examples.

### Training Process
1. **Vocabulary Building**: Count word frequencies and filter rare words
2. **Unigram Table**: Create probability table for efficient negative sampling
3. **Weight Initialization**: Initialize input and output embedding matrices
4. **Training Loop**: For each target-context pair:
   - Compute positive sample loss (context word should have high probability)
   - Sample negative words and compute negative sample loss
   - Update embeddings using gradient descent

## 🔧 Architecture

```
Input Word → Embedding Layer (W1) → Hidden Layer → Output Layer (W2) → Prediction
     ↓                                                      ↑
Target word index                              Context word probabilities
```

- **W1**: Input embeddings matrix (vocab_size × vector_size)
- **W2**: Output embeddings matrix (vocab_size × vector_size)
- **Final embeddings**: Use W1 for word representations

## 📈 Performance

- **Training Speed**: ~9,500 word pairs per second
- **Memory Usage**: O(vocab_size × vector_size)
- **Scalability**: Handles vocabularies up to 1M+ words

### Optimization Tips
- Increase `negative_samples` for better quality (slower training)
- Reduce `min_count` to include more rare words
- Use larger `vector_size` for more nuanced representations
- Adjust `window_size` based on your corpus characteristics

## 📁 File Structure

```
Word2vecfromScratch/
├── word2vec_skipgram.py    # Main implementation
├── README.md               # This file
├── requirements.txt        # Dependencies
├── examples/              # Example usage scripts
└── models/               # Saved models directory
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 👨‍💻 Author

**Monika Chaulagain**
- GitHub: [@MonikaChaulagain](https://github.com/MonikaChaulagain)

## ⭐ Acknowledgments

- Original Word2Vec paper by Mikolov et al.
- Inspired by educational implementations and tutorials
- Built for learning and understanding neural word embeddings


