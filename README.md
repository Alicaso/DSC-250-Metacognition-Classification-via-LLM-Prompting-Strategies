# Metacognition Classification via LLM Prompting Strategies

A comprehensive system for classifying metacognitive behaviors in educational contexts using Large Language Models (LLMs) with advanced prompting techniques including Chain-of-Thought (CoT) reasoning and Retrieval-Augmented Generation (RAG).

## 🎯 Overview

This project implements a sophisticated metacognition classification system based on Zimmerman's self-regulated learning theory. It classifies student reflections into four categories: **Planning**, **Monitoring**, **Evaluating**, and **Not Applicable (NA)**.

## 🚀 Key Features

### LLM Engineering Techniques
- **Multiple Prompting Strategies**: Basic, Few-shot (1/5/10 examples), and Chain-of-Thought prompting
- **Local LLM Deployment**: Uses Ollama with LLaMA 3.2 for privacy and cost efficiency
- **Prompt Engineering**: Sophisticated prompt design with context-aware classification
- **Batch Processing**: Efficient handling of large datasets

### RAG System
- **Vector Database**: FAISS for semantic search over metacognition literature
- **Embedding Model**: SentenceTransformer for text embeddings
- **Knowledge Retrieval**: Contextual information extraction from academic papers

### Classification Pipeline
- **Multi-class Classification**: Four metacognitive categories
- **Performance Evaluation**: Comprehensive metrics (Precision, Recall, F1-Score)
- **Visualization**: Confusion matrices and performance heatmaps
- **Data Generation**: Synthetic dataset creation for training

## 🏗️ Architecture

```
├── ollama_class.py          # Core LLM classification engine
├── bg_context.py           # Context definitions and examples
├── main.ipynb              # Data generation and enhancement pipeline
├── classification.ipynb    # Classification evaluation system
├── rag_system/             # RAG implementation
│   ├── RAG.ipynb          # Vector database and retrieval
│   └── rag.png            # RAG architecture diagram
└── results/               # Experimental results and metrics
    ├── classification/    # Multi-class classification results
    └── classification_monitor_na/  # Binary classification results
```

## 🛠️ Technical Stack

- **LLM Framework**: Ollama (Local LLaMA 3.2)
- **Vector Database**: FAISS
- **Embeddings**: SentenceTransformer
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Development**: Jupyter Notebooks

## 📊 Results

The system achieves impressive performance across different prompting strategies:

- **Planning Classification**: 89.7% Precision, 87.5% Recall
- **Evaluating Classification**: 72.5% Precision, 82.5% Recall
- **Chain-of-Thought**: Enhanced reasoning with detailed explanations
- **RAG Integration**: Context-aware classification with retrieved knowledge

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup Ollama
```bash
# Install Ollama and pull LLaMA 3.2
ollama pull llama3.2
```

### Run Classification
```python
from ollama_class import OllamaLLM

# Initialize classifier
classifier = OllamaLLM()

# Classify text with different strategies
result = classifier.classify_metacognition(
    model="llama3.2",
    text="I will allocate 2 hours daily for studying data structures",
    mode="chain_of_thoughts"
)
```

## 📈 Applications

- **Educational Technology**: Analyze student learning behaviors
- **Learning Analytics**: Track metacognitive development
- **Adaptive Learning**: Personalized learning recommendations
- **Research**: Metacognition studies in CS education

## 🔬 Research Foundation

Based on Zimmerman's Self-Regulated Learning theory:
- **Forethought Phase**: Planning and goal setting
- **Performance Phase**: Monitoring and self-regulation
- **Self-Reflection Phase**: Evaluation and adaptation

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{metacognition_classification,
  title={Metacognition Classification via LLM Prompting Strategies},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/metacognition-classification}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ for advancing educational AI and metacognition research**
