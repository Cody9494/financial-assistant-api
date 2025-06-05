This is a lightweight financial assistant API that answers personal finance questions using an open-source language model.

🔧 Features

Accepts natural language finance questions via a /ask REST endpoint

Uses the unsloth/mistral-7b-instruct-v0.3-bnb-4bit model (CPU/GPU friendly)

Includes basic moderation (content filtering)

Returns a short, helpful answer in JSON format

Getting Started

1. Clone the repository

git clone https://github.com/your-username/genai-finance-assistant.git
cd genai-finance-assistant

2. Install dependencies

pip install fastapi uvicorn torch unsloth

3. Run the app

uvicorn main:app --reload

4. Test it with curl or Postman

curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "How do I start an emergency fund?"}'

Why This Model?

We chose unsloth/mistral-7b-instruct-v0.3-bnb-4bit because:

It’s optimized for CPU/GPU usage with low memory

Instruction-tuned for better responses to user queries

Compact and fast enough to run without needing expensive hardware

Sample Questions to Try

"How can I start budgeting?"

"What's the 50/30/20 rule?"

"How do I save for retirement in my 30s?"

"What's a good credit score?"

"How much emergency savings should I have?"
