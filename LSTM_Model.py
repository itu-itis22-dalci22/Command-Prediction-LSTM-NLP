import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from keras.preprocessing.sequence import pad_sequences
from zemberek import TurkishTokenizer

# Load the dataset
file_path = '/kaggle/input/dronecommands/output.csv'
data = pd.read_csv(file_path, encoding='utf-8')

# Display the first few rows
print(data.head())

commands = data['Command'].values
formations = data[' Label'].values

from zemberek import (
    TurkishMorphology,
    TurkishTokenizer
)
from zemberek import (
    TurkishMorphology,
    TurkishTokenizer
)
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# Initialize Zemberek tokenizer

tokenizer = TurkishTokenizer.DEFAULT

def clean_and_tokenize(text):
    # Convert text to lowercase
    text = text.lower()
    # Tokenize text
    tokens = tokenizer.tokenize(text)
    # Remove punctuation and special characters
    tokens = [token.content for token in tokens if token.content.isalpha()]
    return tokens


commands_cleaned = [clean_and_tokenize(command) for command in commands]

# Split the data into training and testing sets
commands_train, commands_test, formations_train, formations_test = train_test_split(
    commands_cleaned, formations, test_size=0.2, random_state=42)

# Build vocabulary from cleaned training data
vocab = build_vocab_from_iterator(commands_train, specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

# Convert tokenized text to sequences and pad them
commands_seq_train = [torch.tensor(vocab(tokenized_command), dtype=torch.long) for tokenized_command in commands_train]
commands_seq_test = [torch.tensor(vocab(tokenized_command), dtype=torch.long) for tokenized_command in commands_test]

# Padding sequences
commands_seq_train = pad_sequence(commands_seq_train, batch_first=True, padding_value=0)
commands_seq_test = pad_sequence(commands_seq_test, batch_first=True, padding_value=0)

# Encode the formation names
label_encoder = LabelEncoder()
formations_encoded_train = label_encoder.fit_transform(formations_train)
formations_encoded_test = label_encoder.transform(formations_test)

# Prepare the training and testing datasets
class CommandDataset(Dataset):
    def __init__(self, commands, formations):
        self.commands = commands
        self.formations = torch.tensor(formations, dtype=torch.long)
    
    def __len__(self):
        return len(self.commands)
    
    def __getitem__(self, idx):
        return self.commands[idx], self.formations[idx]

train_dataset = CommandDataset(commands_seq_train, formations_encoded_train)
test_dataset = CommandDataset(commands_seq_test, formations_encoded_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

import torch.nn as nn
import torch.optim as optim

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.embedding(x)
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])
        return out

# Hyperparameters
vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 128
output_dim = len(label_encoder.classes_)

# Instantiate the model, define the loss function and the optimizer
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 30
for epoch in range(num_epochs):
    for commands_batch, formations_batch in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(commands_batch)
        
        # Calculate the loss
        loss = criterion(outputs, formations_batch)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

# Save the model
torch.save(model.state_dict(), 'lstm_model.pth')
