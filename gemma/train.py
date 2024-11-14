import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.dataloader import load_training_data, load_eval_data
from src.model_loader import load_model_and_tokenizer
from src.training import train_model
import os

# Load configuration
with open("configs/config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Load dataset
train_dataset = load_training_data(config['dataset']['path'])
eval_dataset = load_eval_data(config['dataset']['path'])

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer(
    config['model']['name']
)

# Train the model
train_model(model, tokenizer, train_dataset, eval_dataset, config)

# Reload the model and tokenizer for inference
model = AutoModelForCausalLM.from_pretrained(config['model']['new_model'])
tokenizer = AutoTokenizer.from_pretrained(config['model']['new_model'])
