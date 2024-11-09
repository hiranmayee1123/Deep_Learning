import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast, AdamW
from torch.utils.data import Dataset, DataLoader
from evaluate import load

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Data loading function
def load_data(file_path):
    with open(file_path, 'rb') as file:
        raw_data = json.load(file)
    
    contexts, questions, answers = [], [], []
    
    for entry in raw_data['data']:
        for paragraph in entry['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    contexts.append(context.lower())
                    questions.append(question.lower())
                    answers.append(answer)
    
    return contexts, questions, answers

# Loading train and validation datasets
train_contexts, train_questions, train_answers = load_data('spoken_train-v1.1.json')
valid_contexts, valid_questions, valid_answers = load_data('spoken_test-v1.1.json')

# Function to calculate end position of answers
def calculate_end_position(answers, contexts):
    for answer, context in zip(answers, contexts):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

calculate_end_position(train_answers, train_contexts)
calculate_end_position(valid_answers, valid_contexts)

# Tokenizer initialization and context truncation
BERT_MAX_LEN = 512
bert_path = "bert-base-uncased"
stride = 128
tokenizer = BertTokenizerFast.from_pretrained(bert_path)

train_contexts_truncated = []

for i in range(len(train_contexts)):
    if len(train_contexts[i]) > BERT_MAX_LEN:
        answer_start = train_answers[i]['answer_start']
        answer_end = train_answers[i]['answer_start'] + len(train_answers[i]['text'])
        midpoint = (answer_start + answer_end) // 2
        context_start = max(0, min(midpoint - BERT_MAX_LEN // 2, len(train_contexts[i]) - BERT_MAX_LEN))
        context_end = context_start + BERT_MAX_LEN
        train_contexts_truncated.append(train_contexts[i][context_start:context_end])
        train_answers[i]['answer_start'] = (BERT_MAX_LEN // 2) - len(train_answers[i]) // 2
    else:
        train_contexts_truncated.append(train_contexts[i])

# Tokenizing inputs
train_encodings = tokenizer(train_questions, train_contexts_truncated, max_length=BERT_MAX_LEN, truncation=True, stride=stride, padding=True)
valid_encodings = tokenizer(valid_questions, valid_contexts, max_length=BERT_MAX_LEN, truncation=True, stride=stride, padding=True)

# Function to find start and end positions of answers
def get_answer_positions(index):
    start_pos, end_pos = 0, 0
    answer_encoding = tokenizer(train_answers[index]['text'], max_length=BERT_MAX_LEN, truncation=True, padding=True)
    for i in range(len(train_encodings['input_ids'][index]) - len(answer_encoding['input_ids'])):
        match = True
        for j in range(1, len(answer_encoding['input_ids']) - 1):
            if answer_encoding['input_ids'][j] != train_encodings['input_ids'][index][i + j]:
                match = False
                break
        if match:
            start_pos = i + 1
            end_pos = i + j + 1
            break
    return start_pos, end_pos

# Generate start and end positions for answers
start_positions = []
end_positions = []
for idx in range(len(train_encodings['input_ids'])):
    s, e = get_answer_positions(idx)
    start_positions.append(s)
    end_positions.append(e)

train_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
valid_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# Dataset class definition
class CustomDataset(Dataset):
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

# Dataset and DataLoader for training and validation
train_dataset = CustomDataset(train_encodings)
valid_dataset = CustomDataset(valid_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1)

# BERT model initialization
bert_model = BertModel.from_pretrained(bert_path)

# Custom model definition
class QuestionAnsweringModel(nn.Module):
    def __init__(self):
        super(QuestionAnsweringModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768 * 2, 768 * 2)
        self.fc2 = nn.Linear(768 * 2, 2)
        self.fc_stack = nn.Sequential(
            self.dropout,
            self.fc1,
            nn.LeakyReLU(),
            self.fc2
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = output[2]
        combined_output = torch.cat((hidden_states[-1], hidden_states[-3]), dim=-1)
        logits = self.fc_stack(combined_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

# Model initialization
model = QuestionAnsweringModel()

# Loss function definition
def compute_loss(start_logits, end_logits, start_positions, end_positions, gamma=1):
    softmax = nn.Softmax(dim=1)
    start_probs = softmax(start_logits)
    inv_start_probs = 1 - start_probs
    end_probs = softmax(end_logits)
    inv_end_probs = 1 - end_probs

    logsoftmax = nn.LogSoftmax(dim=1)
    log_start_probs = logsoftmax(start_logits)
    log_end_probs = logsoftmax(end_logits)

    nll_loss = nn.NLLLoss()

    start_loss = nll_loss(torch.pow(inv_start_probs, gamma) * log_start_probs, start_positions)
    end_loss = nll_loss(torch.pow(inv_end_probs, gamma) * log_end_probs, end_positions)

    return (start_loss + end_loss) / 2

# Optimizer initialization
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=2e-2)

# Training function
def train_model(model, dataloader, epoch):
    model.train()
    losses, acc = [], []
    for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

        loss = compute_loss(start_logits, end_logits, start_positions, end_positions)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        start_pred = torch.argmax(start_logits, dim=1)
        end_pred = torch.argmax(end_logits, dim=1)

        acc.append(((start_pred == start_positions).sum() / len(start_pred)).item())
        acc.append(((end_pred == end_positions).sum() / len(end_pred)).item())

    avg_acc = sum(acc) / len(acc)
    avg_loss = sum(losses) / len(losses)
    return avg_acc, avg_loss

# Evaluation function
def evaluate_model(model, dataloader, threshold=0.5):
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            start_probs = F.softmax(start_logits, dim=1)
            end_probs = F.softmax(end_logits, dim=1)

            start_max_probs, start_pred = torch.max(start_probs, dim=1)
            end_max_probs, end_pred = torch.max(end_probs, dim=1)

            confidence = start_max_probs * end_max_probs

            if confidence >= threshold:
                predicted_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_pred:end_pred]))
                true_answer_start = batch['start_positions'].item()
                true_answer_end = batch['end_positions'].item()
                true_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][true_answer_start:true_answer_end]))
                predictions.append({
                    "Predicted Answer": predicted_answer,
                    "True Answer": true_answer
                })
    return predictions

# Training loop
for epoch in range(1, 6):
    avg_acc, avg_loss = train_model(model, train_loader, epoch)
    print(f"Epoch {epoch}, Loss: {avg_loss}, Accuracy: {avg_acc}")

    predictions = evaluate_model(model, valid_loader)
    print(predictions)
