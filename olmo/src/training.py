from transformers import TrainingArguments
from trl import SFTTrainer

def create_training_args(config):
    return TrainingArguments(
        output_dir=config['training']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        optim=config['training']['optim'],
        adam_beta2=config['training']['adam_beta2'],
        adam_epsilon=config['training']['adam_epsilon'],
        max_grad_norm=config['training']['max_grad_norm'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        learning_rate=config['training']['learning_rate'],
        
        save_steps=config['training']['save_steps'],
        logging_steps=config['training']['logging_steps'],
        
        group_by_length=config['training']['group_by_length'],
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],
        # tf32=config['training']['tf32'],
        
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        gradient_checkpointing=config['training']['gradient_checkpointing'] 
    )

def train_model(model, tokenizer, train_dataset, eval_dataset, config):
    training_args = create_training_args(config)
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="formatted_text",
        # max_seq_length=config['training']['sequence_len'],
        tokenizer=tokenizer,
        args=training_args,
        packing=False
    )
    trainer.train()
    
    # Save the model and the tokenizer
    trainer.model.save_pretrained(config['model']['new_model'])
    tokenizer.save_pretrained(config['model']['new_model'])
