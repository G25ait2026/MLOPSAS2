"""
push_to_hub.py
Push the trained model and tokenizer to the Hugging Face Hub.

Usage:
    python push_to_hub.py --token YOUR_HF_TOKEN --repo YOUR_REPO_NAME
"""

import argparse
from huggingface_hub import login
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import os

def main():
    parser = argparse.ArgumentParser(description="Push model to Hugging Face Hub")
    parser.add_argument("--token", type=str, help="Hugging Face Write Token")
    parser.add_argument("--repo", type=str, help="Repository name (e.g. username/model-name)")
    parser.add_argument("--model_dir", type=str, default="distilbert-goodreads-genres", help="Local directory containing the model")
    
    args = parser.parse_args()
    
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        print("Error: Hugging Face token not provided. Use --token or set HF_TOKEN env var.")
        return

    repo_id = args.repo
    if not repo_id:
        print("Error: Repository ID not provided. Use --repo.")
        return

    print(f"Logging in to Hugging Face Hub...")
    login(token=token)
    
    print(f"Loading model and tokenizer from {args.model_dir}...")
    model = DistilBertForSequenceClassification.from_pretrained(args.model_dir)
    tokenizer = DistilBertTokenizerFast.from_pretrained(args.model_dir)
    
    print(f"Pushing to {repo_id}...")
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    
    print(f"Success! Model pushed to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    main()
