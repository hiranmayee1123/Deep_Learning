import json 
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizerFast, AdamW
from torch.utils.data import Dataset, DataLoader
from evaluate import load
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load dataset function
def load_data(dataset): 
    with open(dataset, 'rb') as f:
        data = json.load(f)
    
    paragraphs = []
    questions = []
    answers = []

    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                for answer in qa['answers']:
                    paragraphs.append(context.lower())
                    questions.append(question.lower())
                    answers.append(answer)
    return paragraphs, questions, answers


# Load the training and validation data
train_contexts, train_questions, train_answers = load_data('spoken_train-v1.1.json')
valid_contexts, valid_questions, valid_answers = load_data('spoken_test-v1.1.json')


# Adjust the answers' end position
def adjust_answer_end(answers, contexts):
    for answer, context in zip(answers, contexts):
        answer['text'] = answer['text'].lower()
        answer['answer_end'] = answer['answer_start'] + len(answer['text'])

adjust_answer_end(train_answers, train_contexts)
adjust_answer_end(valid_answers, valid_contexts)


Bert_Cap = 512
path = "bert-base-uncased"
doc_stride = 128
tokenizer = BertTokenizerFast.from_pretrained(path)
train_contexts_truncated=[]

# Truncate the training contexts that exceed BERT's maximum length
for i in range(len(train_contexts)):
    if len(train_contexts[i]) > 512:
        answer_start = train_answers[i]['answer_start']
        answer_end = train_answers[i]['answer_start'] + len(train_answers[i]['text'])
        mid = (answer_start + answer_end) // 2
        para_start = max(0, min(mid - Bert_Cap // 2, len(train_contexts[i]) - Bert_Cap))
        para_end = para_start + Bert_Cap 
        train_contexts_truncated.append(train_contexts[i][para_start:para_end])
        train_answers[i]['answer_start'] = ((512 // 2) - len(train_answers[i]) // 2)
    else:
        train_contexts_truncated.append(train_contexts[i])

# Tokenize training and validation data
train_encodings = tokenizer(train_questions, train_contexts_truncated, max_length=Bert_Cap, truncation=True, stride=doc_stride, padding=True)
valid_encodings = tokenizer(valid_questions, valid_contexts, max_length=Bert_Cap, truncation=True, stride=doc_stride, padding=True)

# Compute the start and end positions of answers in the tokenized text
def compute_start_and_end(idx):
    start = 0
    end = 0
    answer_encoding = tokenizer(train_answers[idx]['text'], max_length=Bert_Cap, truncation=True, padding=True)
    for a in range(len(train_encodings['input_ids'][idx]) - len(answer_encoding['input_ids'])):
        match = True
        for i in range(1, len(answer_encoding['input_ids']) - 1):
            if train_encodings['input_ids'][idx][a + i] != answer_encoding['input_ids'][i]:
                match = False
                break
        if match:
            start = a + 1
            end = a + i + 1
            break
    return start, end

# Generate start and end positions for each sample
start_positions = []
end_positions = []
for h in range(len(train_encodings['input_ids'])):
    s, e = compute_start_and_end(h)
    start_positions.append(s)
    end_positions.append(e)

# Update the training and validation encodings with the computed positions
train_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})
valid_encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

# Create a dataset class for the model
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

# Create train and validation datasets and data loaders
train_dataset = CustomDataset(train_encodings)
valid_dataset = CustomDataset(valid_encodings)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1)

# Initialize the BERT model
bert_model = BertModel.from_pretrained(path)

# Define the custom model with BERT and a classifier
class QuestionAnsweringModel(nn.Module):
    def __init__(self):
        super(QuestionAnsweringModel, self).__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768 * 2, 768 * 2)
        self.fc2 = nn.Linear(768 * 2, 2)
        self.classifier = nn.Sequential(
            self.dropout,
            self.fc1,
            nn.LeakyReLU(),
            self.fc2 
        )

    def forward(self, input_ids, attention_mask, token_type_ids):
        output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        hidden_states = output[2]
        combined = torch.cat((hidden_states[-1], hidden_states[-3]), dim=-1)
        logits = self.classifier(combined)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        
        return start_logits.squeeze(-1), end_logits.squeeze(-1)

# Initialize the model
model = QuestionAnsweringModel()

# Loss function with focal loss
def focal_loss(start_logits, end_logits, start_positions, end_positions, gamma):
    softmax = nn.Softmax(dim=1)
    probs_start = softmax(start_logits)
    inv_probs_start = 1 - probs_start
    probs_end = softmax(end_logits)
    inv_probs_end = 1 - probs_end
    
    log_softmax = nn.LogSoftmax(dim=1)
    log_probs_start = log_softmax(start_logits)
    log_probs_end = log_softmax(end_logits)
    
    nll = nn.NLLLoss()
    
    focal_start_loss = nll(torch.pow(inv_probs_start, gamma) * log_probs_start, start_positions)
    focal_end_loss = nll(torch.pow(inv_probs_end, gamma) * log_probs_end, end_positions)
    
    return (focal_start_loss + focal_end_loss) / 2

# Optimizer setup
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=2e-2)

# Function to train the model
def train(model, dataloader, epoch):
    model = model.train()
    losses = []
    accuracy = []
    for batch in tqdm(dataloader, desc=f'Epoch {epoch}'):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

        loss = focal_loss(start_logits, end_logits, start_positions, end_positions, gamma=1)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        
        start_pred = torch.argmax(start_logits, dim=1)
        end_pred = torch.argmax(end_logits, dim=1)

        accuracy.append(((start_pred == start_positions).sum() / len(start_pred)).item())
        accuracy.append(((end_pred == end_positions).sum() / len(end_pred)).item())

    return sum(accuracy) / len(accuracy), sum(losses) / len(losses)


# Evaluation function
def evaluate(model, dataloader):
    model = model.eval()
    predictions = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluation'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            start_true = batch['start_positions'].to(device)
            end_true = batch['end_positions'].to(device)
            
            start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

            start_pred = torch.argmax(start_logits)
            end_pred = torch.argmax(end_logits)
            answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_pred:end_pred]))
            true_answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids[0][start_true[0]:end_true[0]]))
            predictions.append([answer, true_answer])

    return predictions

# Word Error Rate (WER) metric
def wer_metric(predictions):
    metric = load('wer')
    wer_score = metric.compute(predictions=predictions)
    return wer_score


# Train the model
for epoch in range(1, 6):
    accuracy, loss = train(model, train_loader, epoch)
    print(f"Epoch {epoch} - Loss: {loss}, Accuracy: {accuracy}")

    predictions = evaluate(model, valid_loader)
    print(wer_metric(predictions))
