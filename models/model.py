import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TrainingArguments, Trainer

def load_model(model_name="deepset/roberta-base-squad2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return model, tokenizer

def fine_tune_qa_model(train_dataset, eval_dataset, model, tokenizer, output_dir="./results"):
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        evaluation_strategy="steps",
        logging_dir='./logs',
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )
    
    trainer.train()
    return model