# Ask a Financial Question – Powered by GenAI (Google Colab Prototype)

This project demonstrates how Generative AI can be used to make personal finance more accessible. It is a prototype that simulates a financial assistant using a large open-source language model.

---

## ✅ What I Built

I created a **Google Colab prototype** that:

- Accepts a **natural language question** (e.g., “How can I start budgeting?”)
- Uses the **`unsloth/mistral-7b-instruct-v0.3-bnb-4bit`** model to generate short, relevant answers
- Constructs chat-style prompts using `chatml` format
- Streams the model’s output in real time

---

## 🧠 Model and Libraries Used

### 🔹 Model: `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
- A quantized, instruction-tuned Mistral variant designed for efficient inference on Colab (GPU).
- Supports 4-bit loading via `unsloth` to avoid out-of-memory errors.

### 🔹 Library: `unsloth`
- Simplifies the loading, tuning, and inference of large transformer models.
- Offers `FastLanguageModel` and chat template utilities to simulate assistant behavior.

### 🔹 Tokenizer Prompting
I applied a structured **chat template** (`chatml`) using:
```python
messages = [
    {"from": "system", "value": "You are a helpful and trustworthy financial assistant."},
    {"from": "user", "value": "How can I start budgeting?"}
]
Then used:

python
Copy
Edit
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
🔹 Output Generation
Used the model’s .generate() function with:

Temperature = 0.1 (low randomness for consistency)

Max tokens = 512

Real-time response via TextStreamer

🧪 Example Output
Input:

How can I start budgeting?

Response:

To start budgeting, follow these steps:

1. Determine your income: Write down all sources of income, including salary, bonuses, and any other regular or irregular income.

2. List your expenses: Write down all your regular expenses, such as rent, utilities, groceries, transportation, and entertainment. Don't forget to include irregular expenses, such as insurance premiums, car maintenance, and holiday expenses.

3. Categorize your expenses: Group your expenses into categories, such as housing, transportation, food, entertainment, and savings.

4. Set financial goals: Decide what you want to achieve financially, such as paying off debt, saving for a down payment on a house, or building an emergency fund.

5. Create a budget: Allocate a specific amount for each expense category based on your income and financial goals. Make sure to leave some room for unexpected expenses.

6. Track your spending: Regularly track your spending to see if you're staying within your budget. Use a budgeting app, spreadsheet, or pen and paper to keep track.

7. Adjust your budget as needed: If you find that you're consistently overspending in a certain category, adjust your budget to reflect more realistic spending habits.

8. Review and adjust your budget regularly: Review your budget regularly to make sure it's still meeting your financial goals. Adjust as needed to account for changes in your income, expenses, or financial goals.

By following these steps, you'll be on your way to creating a budget that helps you manage your money effectively and achieve your financial goals.<|im_end|>


❌ What I Did Not Build (and Why)
🚫 REST API endpoint (/ask)
I focused on the core logic of question-answering.

Did not implement FastAPI due to:

Colab environment not being optimized for REST endpoints

Local system incompatibility with unsloth (requires NVIDIA/Intel GPU)

🚫 Content moderation / jailbreak guardrails
Not implemented in this prototype

Can be added via simple keyword filters or safety classifiers in future versions

🚫 Hosting or Docker setup
This prototype runs entirely in Google Colab using a free GPU session.

🛠️ How to Run the Notebook
Open the Colab notebook

Enable GPU from Runtime > Change runtime type > GPU

Run all cells top to bottom
