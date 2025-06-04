# Financial Assistant API

A lightweight application that uses open-source AI models to answer personal finance questions.

## How to Run Locally

1. Install the required libraries:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
python main.py
```

3. The application will be available at `http://localhost:8000`

## Why These Models?

We have chosen three different models, each with its own advantages:

1. **TinyLlama (1.1B)**
   - Very lightweight (1.1 billion parameters)
   - Ideal for systems with limited resources
   - Good performance in conversational tasks
   - Runs on CPU without special requirements

2. **Phi-2 (2.7B)**
   - Model from Microsoft
   - Excellent answer quality for its size
   - Good performance in reasoning tasks
   - Optimized for conversations

3. **StableLM-2-1.6B**
   - Stable and reliable model from Stability AI
   - Good performance in conversational tasks
   - Lightweight and fast
   - Good community support

## Sample Questions to Try

1. "How can I start budgeting?"
2. "What's the best strategy to pay off my debts?"
3. "How can I start investing in the stock market?"
4. "How can I protect my financial information from scams?"

## Features

- Inappropriate content filtering
- Prompt injection protection
- Clean and concise answers
- Support for multiple models
- Optimized parameters for each model
