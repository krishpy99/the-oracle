from tqdm import tqdm
import numpy as np

def exact_match(prediction, ground_truth):
    return int(prediction.strip().lower() == ground_truth.strip().lower())

def f1_score(prediction, ground_truth):
    pred_tokens = prediction.lower().split()
    truth_tokens = ground_truth.lower().split()
    common = set(pred_tokens) & set(truth_tokens)
    
    if len(common) == 0:
        return 0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)

def evaluate_qa(qa_pipeline, eval_data):
    predictions = []
    for example in tqdm(eval_data):
        result = qa_pipeline(question=example["question"], context=example["context"])
        predictions.append(result["answer"])
    return predictions

def calculate_metrics(predictions, ground_truths):
    em_scores = [exact_match(pred, true) for pred, true in zip(predictions, ground_truths)]
    f1_scores = [f1_score(pred, true) for pred, true in zip(predictions, ground_truths)]
    return np.mean(em_scores), np.mean(f1_scores)
