"""
Generate training data from a large open-source model.

Uses OpenRouter API to access models like DeepSeek, Qwen, Llama, etc.
Generates diverse text that will be converted to Braille for training.

Usage:
    export OPENROUTER_API_KEY="your-key-here"
    python generate_data.py --samples 10000 --output data/generated.jsonl
"""

import os
import json
import argparse
import asyncio
import aiohttp
from pathlib import Path
from typing import List, Dict
import random

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Diverse prompts to generate varied training data
GENERATION_PROMPTS = [
    "Write a short paragraph about nature.",
    "Explain a simple science concept for children.",
    "Tell a brief story with a moral lesson.",
    "Describe an everyday activity step by step.",
    "Write a friendly conversation between two people.",
    "Explain how something works in simple terms.",
    "Write a short poem about friendship.",
    "Describe a place you might visit on vacation.",
    "Write instructions for a simple task.",
    "Tell a fun fact and explain why it's interesting.",
    "Write a short dialogue between a teacher and student.",
    "Describe what you see when you look outside.",
    "Write a brief news article about a positive event.",
    "Explain a hobby and why someone might enjoy it.",
    "Write a thank you note to someone helpful.",
    "Describe your favorite food and how it's made.",
    "Write a short adventure story.",
    "Explain an emotion and when someone might feel it.",
    "Write a letter to a friend about your day.",
    "Describe an animal and its interesting behaviors.",
]

# Models available via OpenRouter (sorted by quality/cost tradeoff)
MODELS = {
    "deepseek": "deepseek/deepseek-chat",
    "qwen": "qwen/qwen-2.5-72b-instruct",
    "llama": "meta-llama/llama-3.1-70b-instruct",
    "mistral": "mistralai/mistral-large-2411",
}


async def generate_single(
    session: aiohttp.ClientSession,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int = 200,
) -> str:
    """Generate a single text sample from the API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/braille-frontier",
    }
    
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant. Write clear, simple text suitable for all ages. Keep responses concise and educational."
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": max_tokens,
        "temperature": 0.9,  # High temperature for diversity
    }
    
    try:
        async with session.post(OPENROUTER_API_URL, headers=headers, json=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                return data["choices"][0]["message"]["content"]
            else:
                error = await resp.text()
                print(f"API error {resp.status}: {error[:100]}")
                return None
    except Exception as e:
        print(f"Request error: {e}")
        return None


async def generate_batch(
    api_key: str,
    model: str,
    n_samples: int,
    output_path: Path,
    batch_size: int = 10,
) -> int:
    """Generate a batch of training samples."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    generated = 0
    
    async with aiohttp.ClientSession() as session:
        with open(output_path, "w") as f:
            for i in range(0, n_samples, batch_size):
                # Create batch of prompts
                prompts = [random.choice(GENERATION_PROMPTS) for _ in range(batch_size)]
                
                # Generate in parallel
                tasks = [
                    generate_single(session, api_key, model, prompt)
                    for prompt in prompts
                ]
                results = await asyncio.gather(*tasks)
                
                # Save successful generations
                for prompt, text in zip(prompts, results):
                    if text:
                        record = {
                            "prompt": prompt,
                            "text": text.strip(),
                            "model": model,
                        }
                        f.write(json.dumps(record) + "\n")
                        generated += 1
                
                print(f"Generated {generated}/{n_samples} samples...")
                
                # Rate limiting
                await asyncio.sleep(0.5)
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate training data from LLM")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="data/generated.jsonl", help="Output file path")
    parser.add_argument("--model", type=str, default="deepseek", choices=list(MODELS.keys()), help="Model to use")
    args = parser.parse_args()
    
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("Error: OPENROUTER_API_KEY environment variable not set")
        print("\nTo get an API key:")
        print("1. Go to https://openrouter.ai/")
        print("2. Sign up and add credits")
        print("3. Create an API key")
        print("4. Run: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    model = MODELS[args.model]
    print(f"Generating {args.samples} samples using {model}...")
    
    n_generated = asyncio.run(generate_batch(
        api_key=api_key,
        model=model,
        n_samples=args.samples,
        output_path=Path(args.output),
    ))
    
    print(f"\nDone! Generated {n_generated} samples to {args.output}")


if __name__ == "__main__":
    main()
