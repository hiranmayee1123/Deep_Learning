import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast, AdamW
from torch.utils.data import Dataset, DataLoader
from evaluate import load

# Set device for GPU if available
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load dataset from JSON file
def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = json.load(f)
    
    contexts, questions, answers = [], [], []

    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context.lower())
                    questions.append(question.lower())
                    answers.append(answer)
    
    return contexts, questions, answers

# Load train and validation datasets
train_contexts, train_questions, train_answers = load_data('spoken_train-v1.1.json')
valid_contexts, valid_questions, valid_answers = load_data('spoken_test-v1.1.json')

# Adjust answer end positions
def calculate_answer_end(answers, contexts):
    for answer, context in zip(answers, contexts):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

calculate_answer_end(train_answers, train_contexts)
calculate_answer_end(valid_answers, valid_contexts)

# Tokenizer setup for BERT
BERT_MAX_LENGTH = 512
bert_path = "bert-base-uncased"
doc_stride = 256
tokenizer = BertTokenizerFast.from_pretrained(bert_path)

# Encode train and validation data
train_encodings = tokenizer(train_questions, train_contexts, max_length=BERT_MAX_LENGTH, truncation=True, stride=doc_stride, padding=True)
valid_encodings = tokenizer(valid_questions, valid_contexts, max_length=BERT_MAX_LENGTH, truncation=True, stride=doc_stride, padding=True)

# Define start and end positions for answers
def get_answer_positions(idx, encodings, answers):
    ret_start, ret_end = 0, 0
    answer_encoding = tokenizer(answers[idx]['text'], max_length=BERT_MAX_LENGTH, truncation=True, padding=True)
    
    for a in range(len(encodings['input_ids'][idx]) - len(answer_encoding['input_ids'])):
        match = True
        for i in range(1, len(answer_encoding['input_ids']) - 1):
            if encodings['input_ids'][idx][a + i] != answer_encoding['input_ids'][i]:
                match = False
                break
        if match:
            ret_start, ret_end = a + 1, a + i + 1
            break
    return ret_start, ret_end

# Calculate start and end positions for the train set
train_start_positions, train_end_positions = [], []
for i in range(len(train_encodings['input_ids'])):
    start, end = get_answer_positions(i, train_encodings, train_answers)
    train_start_positions.append(start)
    train_end_positions.append(end)

train_encodings.update({'start_positions': train_start_positions, 'end_positions': train_end_positions})

# Calculate start and end positions for the validation set
valid_start_positions, valid_end_positions = [], []
for i in range(len(valid_encodings['input_ids'])):
    start, end = get_answer_positions(i, valid_encodings, valid_answers)
    valid_start_positions.append(start)
    valid_end_positions.append(end)

valid_encodings.update({'start_positions': valid_start_positions, 'end_positions': valid_end_positions})

# Dataset class for input data
class QAInputDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
    
    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'token_type_ids': torch.tensor(self.encodings['token_type_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'start_positions': torch.tensor(self.encodings['start_positions'][idx]),
            'end_positions': torch.tensor(self.encodings['end_positions'][idx])
        }
    
    def __len__(self):
        return len(self.encodings['input_ids'])

# Initialize data loaders
train_dataset = QAInputDataset(train_encodings)
valid_dataset = QAInputDataset(valid_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1)

# Load pre-trained BERT model
bert_model = BertModel.from_pretrained(bert_path)

# Define the new model
class QuestionAnsweringModel(nn.Module):
    def __init__(self):
        super(QuestionAnsweringModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768 * 2, 768 * 2)
        self.fc2 = nn.Linear(768 * 2, 2)
        self.fc_stack = nn.Sequential(self.dropout, self.fc1, nn.LeakyReLU(), self.fc2)
    
    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = output[2]
        combined_states = torch.cat((hidden_states[-1], hidden_states[-3]), dim=-1)
        logits = self.fc_stack(combined_states)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

# Initialize the model
model = QuestionAnsweringModel()

# Loss function
def compute_loss(start_logits, end_logits, start_positions, end_positions, gamma):
    softmax = nn.Softmax(dim=1)
    start_probs = softmax(start_logits)
    end_probs = softmax(end_logits)
    
    inverse_start_probs = 1 - start_probs
    inverse_end_probs = 1 - end_probs
    
    log_softmax = nn.LogSoftmax(dim=1)
    log_start_probs = log_softmax(start_logits)
    log_end_probs = log_softmax(end_logits)
    
    nll_loss = nn.NLLLoss()
    
    fl_start = nll_loss(torch.pow(inverse_start_probs, gamma) * log_start_probs, start_positions)
    fl_end = nll_loss(torch.pow(inverse_end_probs, gamma) * log_end_probs, end_positions)
    
    return (fl_start + fl_end) / 2

# Optimizer
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=2e-2)

# Training function
def train(model, dataloader, epoch):
    model.train()
    total_loss = []
    total_acc = []
    batch_counter = 0
    for batch in tqdm(dataloader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        
        # Get inputs and targets from the batch
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)
        
        # Forward pass
        start_preds, end_preds = model(input_ids, attention_mask, token_type_ids)

        # Compute loss
        loss = compute_loss(start_preds, end_preds, start_positions, end_positions, gamma=1)
        total_loss.append(loss.item())
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        start_pred = torch.argmax(start_preds, dim=1)
        end_pred = torch.argmax(end_preds, dim=1)
        
        start_acc = (start_pred == start_positions).sum() / len(start_pred)
        end_acc = (end_pred == end_positions).sum() / len(end_pred)
        
        total_acc.append((start_acc + end_acc) / 2)
        batch_counter += 1
        
    avg_loss = sum(total_loss) / len(total_loss)
    avg_acc = sum(total_acc) / len(total_acc)
    return avg_acc, avg_loss

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    answer_list = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            
            start_preds, end_preds = model(input_ids, attention_mask, token_type_ids)
            
            start_pred = torch.argmax(start_preds)
            end_pred = torch.argmax(end_preds)
            
            pred_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_pred:end_pred]))
            true_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_true[0]:end_true[0]]))
            
            answer_list.append([pred_answer, true_answer])

    return answer_list

# Load Word Error Rate (WER) metric for evaluation
wer_metric = load("wer")

# Move model to device (GPU or CPU)
model.to(device)
