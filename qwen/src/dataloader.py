from datasets import load_dataset, DatasetDict

def load_data_with_split(dataset_path, test_size=0.1, val_size=0.1, seed=42):

    dataset = load_dataset('csv', data_files=dataset_path, split='train')

    train_testvalid = dataset.train_test_split(test_size=test_size + val_size, seed=seed)

    test_valid = train_testvalid['test'].train_test_split(test_size=test_size / (test_size + val_size), seed=seed)

    dataset_dict = DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })

    return dataset_dict

#dataformat specific for qwen
def format_data(row):
    return {
        'formatted_text': f"<s> [INST] {row['input']} [/INST] {row['output']} </s>"
    }

def load_and_format_data(dataset_path, test_size=0.1, val_size=0.1, seed=42):
    dataset_dict = load_data_with_split(dataset_path, test_size, val_size, seed)
    
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].map(format_data)
    
    return dataset_dict

def load_training_data(dataset_path, test_size=0.1, val_size=0.1, seed=42):
    dataset_dict = load_and_format_data(dataset_path, test_size, val_size, seed)
    return dataset_dict['train']

def load_eval_data(dataset_path, test_size=0.1, val_size=0.1, seed=42):
    dataset_dict = load_and_format_data(dataset_path, test_size, val_size, seed)
    return dataset_dict['validation']

def load_test_data(dataset_path, test_size=0.1, val_size=0.1, seed=42):
    dataset_dict = load_and_format_data(dataset_path, test_size, val_size, seed)
    return dataset_dict['test']