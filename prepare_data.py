import json
from datasets import Dataset, DatasetDict
import random

def load_qa_pairs(file_path: str):
    """Load QA pairs from generated json file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['qa_pairs']

def format_for_training(qa_pair: dict) -> dict:
    """Format QA pair into prompt and completion format"""
    # Format question
    question = qa_pair['question']
    
    # Join answer steps into executable code block
    answer_code = '\n'.join(qa_pair['answer_steps'])
    
    # Create prompt with instruction
    prompt = f"Given a pandas DataFrame 'df' loaded with SEC statistics, answer the following question:\n\nQuestion: {question}\n\nProvide Python code to answer this question."
    
    return {
        'prompt': prompt,
        'completion': answer_code,
        'category': qa_pair['category'],
        'answer_type': qa_pair['answer_type']
    }

def split_data(data, train_ratio=0.5, val_ratio=0.2):
    """Split data into train/val/test sets"""
    random.shuffle(data)
    n = len(data)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)
    
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    return train_data, val_data, test_data

def main():
    # Load QA pairs
    qa_pairs = load_qa_pairs('generated_qa_pairs.json')
    
    # Format data for training
    formatted_data = [format_for_training(qa) for qa in qa_pairs]
    
    # Split data
    train_data, val_data, test_data = split_data(formatted_data)
    
    # Create dataset dictionary
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'validation': Dataset.from_list(val_data),
        'test': Dataset.from_list(test_data)
    })
    
    # Save to jsonl files
    dataset_dict.save_to_disk('sec_qa_dataset')
    
    # Also save as jsonl
    for split in ['train', 'validation', 'test']:
        dataset_dict[split].to_json(f'sec_qa_{split}.jsonl')
    
    print(f"Total examples: {len(formatted_data)}")
    print(f"Train size: {len(train_data)}")
    print(f"Validation size: {len(val_data)}")
    print(f"Test size: {len(test_data)}")

if __name__ == "__main__":
    main() 