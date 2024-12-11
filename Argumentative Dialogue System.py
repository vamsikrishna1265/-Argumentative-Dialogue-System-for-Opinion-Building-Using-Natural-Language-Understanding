import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer, RobertaModel, get_scheduler
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, pearsonr
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Load both datasets
banking77_dataset = load_dataset('banking77')
sts_dataset = load_dataset('stsb_multi_mt', 'en', split='train')

# Preprocessing
MAX_LEN = 128
BATCH_SIZE = 16
HIDDEN_DIM = 512  # Increased hidden dimension for BiLSTM
OUTPUT_DIM = 77
DROPOUT_PROB = 0.3
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Dataset class for Banking77
class BankingDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Split dataset into train and validation sets for Banking77
train_texts, val_texts, train_labels, val_labels = train_test_split(
    banking77_dataset['train']['text'], banking77_dataset['train']['label'], test_size=0.2
)

train_dataset = BankingDataset(train_texts, train_labels, tokenizer, MAX_LEN)
val_dataset = BankingDataset(val_texts, val_labels, tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Define BiLSTM + RoBERTa model for intent classification
class IntentClassifier(nn.Module):
    def __init__(self, bert, hidden_dim, output_dim, dropout_prob=0.3):
        super(IntentClassifier, self).__init__()
        self.bert = bert
        self.lstm = nn.LSTM(bert.config.hidden_size, hidden_dim, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        lstm_out, _ = self.lstm(last_hidden_state)
        lstm_out = self.dropout(lstm_out[:, -1, :])  # Get the last output for classification
        return self.fc(lstm_out)

# Instantiate the Intent Classifier model
intent_classifier_model = IntentClassifier(RobertaModel.from_pretrained('roberta-base'), HIDDEN_DIM, OUTPUT_DIM, DROPOUT_PROB)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(intent_classifier_model.parameters(), lr=2e-5)
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 5)

# Training loop for Banking77 intent classifier
for epoch in range(5):
    intent_classifier_model.train()
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['label']

        optimizer.zero_grad()
        outputs = intent_classifier_model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

# Define SBERT-style Sentence Similarity Model for STS Benchmark
class SimilarityModel(nn.Module):
    def __init__(self, bert):
        super(SimilarityModel, self).__init__()
        self.bert = bert

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

# Instantiate and use the similarity model
similarity_model = SimilarityModel(RobertaModel.from_pretrained('roberta-base'))

# Precompute sentence embeddings for STS Benchmark
def compute_similarity(user_input, templates):
    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=128)
    user_embedding = similarity_model(inputs['input_ids'], inputs['attention_mask'])

    similarities = []
    for template in templates:
        template_input = tokenizer(template, return_tensors='pt', padding=True, truncation=True, max_length=128)
        template_embedding = similarity_model(template_input['input_ids'], template_input['attention_mask'])
        similarity = F.cosine_similarity(user_embedding, template_embedding)
        similarities.append(similarity.item())
    
    return similarities

# Example use for the argumentative templates
argumentative_templates = [
    "That's an interesting claim! Could you provide some supporting evidence?",
    "I see your point, but how would you address the opposing view?",
    "Interesting claim. How do you think this could be implemented effectively?",
    "That sounds like solid reasoning! Can you elaborate on your evidence?",
    "That's a valid perspective, but have you considered the challenges it might face?"
]

# Evaluate on STS Benchmark
def evaluate_similarity():
    y_true = []
    y_pred = []
    for sample in sts_dataset:
        input_text = sample['sentence1']
        target_text = sample['sentence2']
        true_score = sample['score']
        
        # Get predicted similarity
        similarities = compute_similarity(input_text, argumentative_templates)
        predicted_score = np.max(similarities)  # For simplicity, just taking max similarity
        
        y_true.append(true_score)
        y_pred.append(predicted_score)
    
    # Compute Pearson correlation
    pearson_corr, _ = pearsonr(y_true, y_pred)
    print(f"Pearson correlation for STS Benchmark: {pearson_corr}")

# Combined System
def unified_response_system(user_input):
    inputs = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True, max_length=128)

    # Intent classification
    intent_classifier_model.eval()
    with torch.no_grad():
        intent_outputs = intent_classifier_model(inputs['input_ids'], inputs['attention_mask'])
        predicted_intent = torch.argmax(intent_outputs, dim=1).item()

    intent_responses = {
        0: "Sure, I can help with balance inquiries.",
        1: "Let's look into your transaction history.",
    }

    if predicted_intent in intent_responses:
        return intent_responses[predicted_intent]

    # Argumentative similarity
    similarities = compute_similarity(user_input, argumentative_templates)
    most_similar_idx = np.argmax(similarities)

    return argumentative_templates[most_similar_idx]

# Example usage
if __name__ == "__main__":
    user_input = "I believe that renewable energy should replace fossil fuels."
    response = unified_response_system(user_input)
    print(f"User: {user_input}")
    print(f"System: {response}")

    # Evaluate STS Benchmark
    evaluate_similarity()
