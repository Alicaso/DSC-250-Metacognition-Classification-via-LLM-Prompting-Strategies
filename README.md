# Metacognition Classification via LLM Prompting Strategies

A system for classifying metacognitive behaviors in educational contexts using Large Language Models with advanced prompting techniques including Chain-of-Thought reasoning and Retrieval-Augmented Generation.

## Overview

This project implements a metacognition classification system based on Zimmerman's self-regulated learning theory. It classifies student reflections into four categories: **Planning**, **Monitoring**, **Evaluating**, and **Not Applicable (NA)**.

## Key Features

- **Multiple Prompting Strategies**: Basic, Few-shot (1/5/10 examples), and Chain-of-Thought prompting
- **Local LLM Deployment**: Uses Ollama with LLaMA 3.2
- **RAG System**: FAISS vector database with SentenceTransformer embeddings
- **Multi-class Classification**: Four metacognitive categories with comprehensive evaluation metrics
- **Data Generation**: Synthetic dataset creation and real data enhancement
- **Visualization**: Confusion matrices and performance heatmaps

## Results

- **Planning Classification**: 89.7% Precision, 87.5% Recall
- **Evaluating Classification**: 72.5% Precision, 82.5% Recall
- **Chain-of-Thought**: Enhanced reasoning with detailed explanations
- **RAG Integration**: Context-aware classification with retrieved knowledge

## Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Setup Ollama
```bash
ollama pull llama3.2
```

### Run Classification
```python
from ollama_class import OllamaLLM

classifier = OllamaLLM()
result = classifier.classify_metacognition(
    model="llama3.2",
    text="I will allocate 2 hours daily for studying data structures",
    mode="chain_of_thoughts"
)
```

## Applications

- **Educational Technology**: Analyze student learning behaviors
- **Learning Analytics**: Track metacognitive development
- **Adaptive Learning**: Personalized learning recommendations
- **Research**: Metacognition studies in CS education
