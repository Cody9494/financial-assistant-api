from fastapi import FastAPI, HTTPException, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import uvicorn
import logging
import re
from typing import List, Dict
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Financial Assistant API")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

import string

# Less aggressive BLOCKED_WORDS
BLOCKED_WORDS = [
    "ignore previous", "jailbreak", "system prompt", "act as", "pretend", "bypass filter",
    "simulate", "disregard above"
]

# Truly harmful topics (still blocked)
BLOCKED_TOPICS = [
    "create virus", "how to scam", "buy stolen data", "sell drugs", "commit fraud", "dark web"
]

# Allowed financial-sensitive topics (if context is safe)
FINANCIAL_SENSITIVE_TOPICS = {
    "scam": ["avoid", "detect", "report", "signs", "victim", "protect", "prevent"],
    "fraud": ["prevent", "detect", "report", "tax", "identity", "protect", "avoid"],
    "phishing": ["identify", "avoid", "report", "protect", "prevent"],
    "debt": ["manage", "reduce", "consolidate", "plan"],
    "bankruptcy": ["options", "legal", "consequences", "rebuild"],
    "hacking": ["protect", "secure", "avoid"],
    "crypto": ["invest", "risks", "wallet", "safe", "scams"],
}


# Available models configuration
AVAILABLE_MODELS = {
    "tinyllama": {
        "name": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "prompt_template": "<|system|>You are a helpful financial assistant. Provide clear, practical advice.</s><|user|>{question}</s><|assistant|>"
    },
    "phi2": {
        "name": "microsoft/phi-2",
        "prompt_template": "Instruct: You are a helpful financial assistant. Provide clear, practical advice.\n\nInput: {question}\n\nOutput:"
    },
    "stablelm": {
        "name": "stabilityai/stablelm-2-1_6b",
        "prompt_template": "<|system|>You are a helpful financial assistant. Provide clear, practical advice.</s><|user|>{question}</s><|assistant|>"
    }
}

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    model_choice: str = Field(..., pattern="^(tinyllama|phi2|stablelm)$")

def check_content_safety(text: str) -> bool:
    """Check if the text contains any blocked words or topics with context awareness."""
    text_lower = text.lower().translate(str.maketrans('', '', string.punctuation))
    
    # First check for system prompt injection attempts
    if any(word in text_lower for word in BLOCKED_WORDS):
        return False
    
    # Check for security-related topics with context
    for topic, safe_terms in FINANCIAL_SENSITIVE_TOPICS.items():
        if topic in text_lower:
            # If the topic is mentioned, check if it's in a safe context
            if any(safe_word in text_lower for safe_word in safe_terms):
                return True  # ✅ ALLOW: It's in a safe context
            else:
                return False  # ❌ BLOCK: Detected misuse of term
    
    # For other topics, check blocked topics
    return not any(topic in text_lower for topic in BLOCKED_TOPICS)


def sanitize_prompt(prompt: str) -> str:
    """Remove any potentially harmful content from the prompt."""
    # Remove any attempts to override system instructions
    prompt = re.sub(r'(?i)(ignore|override|system|prompt|instructions).*?(above|below|all|everything)', '', prompt)
    
    # Remove any attempts to inject code
    prompt = re.sub(r'```.*?```', '', prompt, flags=re.DOTALL)
    
    # Clean up any extra whitespace
    prompt = ' '.join(prompt.split())
    
    return prompt.strip()

# Load models and tokenizers
models: Dict[str, Dict] = {}
tokenizers: Dict[str, AutoTokenizer] = {}

for model_id, config in AVAILABLE_MODELS.items():
    try:
        tokenizers[model_id] = AutoTokenizer.from_pretrained(config["name"])
        models[model_id] = AutoModelForCausalLM.from_pretrained(
            config["name"],
            torch_dtype=torch.float16,
            device_map="auto"
        )
        logger.info(f"Successfully loaded model: {config['name']}")
    except Exception as e:
        logger.error(f"Failed to load model {config['name']}: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    logger.info("Serving home page")
    return templates.TemplateResponse("index.html", {
        "request": request,
        "title": "Financial Assistant",
        "description": "Ask me anything about personal finance and budgeting",
        "placeholder": "Type your question here...",
        "submit_text": "Get Advice",
        "models": list(AVAILABLE_MODELS.keys())
    })

def clean_response(response: str) -> str:
    """Clean the model's response from unwanted content."""
    # Remove HTML tags
    response = re.sub(r'<[^>]+>', '', response)
    # Remove system and user tokens
    response = re.sub(r'<\|system\|>.*?<\|assistant\|>', '', response, flags=re.DOTALL)
    response = re.sub(r'<\|user\|>.*?<\|assistant\|>', '', response, flags=re.DOTALL)
    # Remove any remaining tokens
    response = re.sub(r'<\|.*?\|>', '', response)
    # Remove multiple newlines
    response = re.sub(r'\n\s*\n', '\n\n', response)
    # Remove any URLs
    response = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', response)
    return response.strip()

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Content safety check
        if not check_content_safety(request.question):
            raise HTTPException(
                status_code=400,
                detail="Your question contains inappropriate content. Please rephrase it."
            )
        
        # Sanitize the prompt
        sanitized_question = sanitize_prompt(request.question)
        
        # Get model configuration
        model_config = AVAILABLE_MODELS[request.model_choice]
        model = models[request.model_choice]
        tokenizer = tokenizers[request.model_choice]
        
        prompt = model_config["prompt_template"].format(question=sanitized_question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Adjust generation parameters based on model
        generation_params = {
            "max_new_tokens": 512,
            "do_sample": True,
            "repetition_penalty": 1.2,
            "no_repeat_ngram_size": 3
        }
        
        if request.model_choice == "phi2":
            generation_params.update({
                "temperature": 0.8,
                "top_p": 0.95,
                "top_k": 50
            })
        elif request.model_choice == "stablelm":
            generation_params.update({
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            })
        else:  # tinyllama
            generation_params.update({
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40
            })
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                **generation_params
            )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace(prompt, "").strip()
        response = clean_response(response)
        
        # Final content safety check on response
        if not check_content_safety(response):
            raise HTTPException(
                status_code=500,
                detail="Generated response contains inappropriate content. Please try again."
            )
        
        return {"answer": response}
    
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
