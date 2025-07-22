# 📘 PRODIGY INFOTECH – Task 03  
## 🎯 Text Generation with Markov Chains  

This project demonstrates how to build a **Markov Chain-based text generator** that learns word transitions from a given corpus and generates new sentences based on learned probabilities.

---

## 🔍 Overview

This task involves:
- 🧮 Building a probabilistic language model using Markov chains
- 📖 Training it on real-world data (Shakespearean texts)
- 🤖 Generating new sentences based on learned transitions
- 📊 Visualizing common word associations

The implementation supports configurable **n-gram order** and provides both seeded and random sentence generation capabilities.

---

## 💻 Running the Project

### Prerequisites
- Google Colab or local Python environment
- Internet access for dataset downloads

### Steps
1. Open `PRODIGY_GA_03.ipynb` in Colab.
2. Run all cells sequentially.
3. Observe the generated text and visualizations.

---

## 💡 Key Takeaways

- Markov chains offer a simple yet effective way to model sequence dependencies.
- While basic models can't compete with modern transformer-based systems, they serve as a great introduction to language modeling concepts.
- With extensions like POS tagging and smoothing, these models can become surprisingly expressive.

---

## 🧩 Future Enhancements

- Add part-of-speech awareness
- Implement transition smoothing
- Support character-level generation
- Build an API around the model
