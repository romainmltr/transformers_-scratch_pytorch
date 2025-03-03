import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import time
import os

# Set random seeds for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Charger le texte depuis le fichier local
print("Chargement des données...")
try:
    with open('tiny_shakespeare.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"Données chargées avec succès: {len(text)} caractères")
except FileNotFoundError:
    print("Erreur: Fichier tiny_shakespeare.txt non trouvé.")
    print("Vérifiez que le fichier est dans le même répertoire que ce script.")
    exit(1)

# Diviser les données en ensembles d'entraînement et de test (90% / 10%)
train_size = int(len(text) * 0.9)
train_text = text[:train_size]
test_text = text[train_size:]

print(f"Texte d'entraînement: {len(train_text)} caractères")
print(f"Texte de test: {len(test_text)} caractères")

# Create character-level tokenizer
class CharTokenizer:
    def __init__(self, text):
        # Get unique characters from text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create mappings from characters to indices and vice versa
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        
        print(f"Vocabulaire: {self.vocab_size} caractères uniques")
    
    def encode(self, string):
        """Convert a string to a list of integers."""
        return [self.char_to_idx[ch] for ch in string]
    
    def decode(self, indices):
        """Convert a list of integers to a string."""
        return ''.join([self.idx_to_char[idx] for idx in indices])
    
    def encode_one_hot(self, string):
        """Convert a string to a one-hot encoded tensor."""
        indices = self.encode(string)
        return F.one_hot(torch.tensor(indices), num_classes=self.vocab_size).float()

# Initialize the tokenizer
tokenizer = CharTokenizer(text)

# Create a character-level dataset
class ShakespeareDataset(Dataset):
    def __init__(self, text, seq_length, tokenizer):
        self.text = text
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.data = tokenizer.encode(text)
        self.total_seq = len(self.data) - seq_length
    
    def __len__(self):
        return self.total_seq
    
    def __getitem__(self, idx):
        # Get input sequence and target sequence
        x = self.data[idx:idx+self.seq_length]
        y = self.data[idx+1:idx+self.seq_length+1]  # Shifted by 1 position
        
        return torch.tensor(x), torch.tensor(y)

# Create the RNN model
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # RNN layers
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, hidden=None):
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=x.device)
        
        # Convert input to embeddings
        x = self.embedding(x)
        
        # Pass through RNN
        out, hidden = self.rnn(x, hidden)
        
        # Reshape for fully connected layer
        out = out.reshape(-1, self.hidden_size)
        
        # Pass through fully connected layer
        out = self.fc(out)
        
        return out, hidden
    
    def generate(self, tokenizer, seed_text, max_length=100, temperature=1.0):
        """Generate text starting with seed_text."""
        self.eval()  # Set to evaluation mode
        
        with torch.no_grad():
            # Convert seed text to indices
            current_indices = tokenizer.encode(seed_text)
            result = seed_text
            hidden = None
            
            # Generate one character at a time
            for _ in range(max_length):
                # Convert to tensor and add batch dimension
                x = torch.tensor([current_indices[-1]]).unsqueeze(0)
                
                # Predict next character
                output, hidden = self(x, hidden)
                
                # Apply temperature to adjust randomness
                output = output / temperature
                
                # Convert to probabilities
                probs = F.softmax(output, dim=1).squeeze()
                
                # Sample from the distribution
                next_index = torch.multinomial(probs, 1).item()
                
                # Add predicted character to result
                result += tokenizer.idx_to_char[next_index]
                current_indices.append(next_index)
            
            return result

# Set hyperparameters
seq_length = 100
hidden_size = 128
num_layers = 2
batch_size = 64
learning_rate = 0.001
num_epochs = 3  # Diminué pour accélérer le test

# Create the datasets and dataloaders
train_dataset = ShakespeareDataset(train_text, seq_length, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = ShakespeareDataset(test_text, seq_length, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = RNN(tokenizer.vocab_size, hidden_size, tokenizer.vocab_size, num_layers)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Function to calculate perplexity
def calculate_perplexity(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    total_count = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs, _ = model(inputs)
            loss = criterion(outputs, targets.view(-1))
            total_loss += loss.item() * inputs.size(0) * seq_length
            total_count += inputs.size(0) * seq_length
    
    avg_loss = total_loss / total_count
    perplexity = math.exp(avg_loss)
    
    return avg_loss, perplexity

# Training loop with validation
def train_and_validate(model, train_loader, test_loader, criterion, optimizer, epochs):
    model.train()
    best_perplexity = float('inf')
    train_losses = []
    test_losses = []
    test_perplexities = []
    
    print("\n=== DÉBUT DE L'ENTRAÎNEMENT ===")
    for epoch in range(epochs):
        epoch_start_time = time.time()
        total_train_loss = 0
        
        # Training phase
        model.train()
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            # Forward pass
            outputs, _ = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, targets.view(-1))
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Print progress
            if (batch_idx + 1) % 100 == 0:
                print(f'Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Testing phase
        avg_test_loss, test_perplexity = calculate_perplexity(model, test_loader, criterion)
        test_losses.append(avg_test_loss)
        test_perplexities.append(test_perplexity)
        
        # Print epoch statistics
        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch+1}/{epochs} - {epoch_time:.1f}s - '
              f'Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, '
              f'Test Perplexity: {test_perplexity:.4f}')
        
        # Generate sample text
        seed_text = "ROMEO: "
        generated_text = model.generate(tokenizer, seed_text, max_length=100)
        print(f"\nExemple de texte généré:\n{generated_text}\n")
        
        # Save best model
        if test_perplexity < best_perplexity:
            best_perplexity = test_perplexity
            torch.save(model.state_dict(), 'shakespeare_rnn_best1.pth')
            print(f"Nouveau meilleur modèle enregistré avec perplexité: {best_perplexity:.4f}")
    
    print("\n=== ENTRAÎNEMENT TERMINÉ ===")
    return train_losses, test_losses, test_perplexities

# Train the model
train_losses, test_losses, test_perplexities = train_and_validate(
    model, train_loader, test_loader, criterion, optimizer, num_epochs
)

# Save the final model
torch.save(model.state_dict(), 'shakespeare_rnn_final.pth')
print("Modèle final enregistré sous 'shakespeare_rnn_final.pth'")

# Load the best model
try:
    model.load_state_dict(torch.load('shakespeare_rnn_best1.pth'))
    print("Meilleur modèle chargé pour les tests")
except:
    print("Utilisation du modèle final pour les tests")

# Test the model with different seeds and temperatures
def test_generation(model, tokenizer, seed_texts, temperatures):
    print("\n=== TESTS DE GÉNÉRATION DE TEXTE ===")
    for seed in seed_texts:
        print(f"\n--- Amorce: \"{seed}\" ---")
        for temp in temperatures:
            generated = model.generate(tokenizer, seed, max_length=200, temperature=temp)
            print(f"\nTempérature: {temp}")
            print(generated)
            print("-" * 50)

# Test seeds and temperatures
test_seeds = [
    "ROMEO: ",
    "JULIET: ",
    "HAMLET: ",
    "To be or not to be",
    "All the world's a stage"
]

test_temperatures = [0.5, 0.8, 1.2]

# Run the generation tests
model.eval()
test_generation(model, tokenizer, test_seeds, test_temperatures)

# Compute final evaluation metrics
print("\n=== ÉVALUATION FINALE ===")
final_loss, final_perplexity = calculate_perplexity(model, test_loader, criterion)
print(f"Perte finale sur l'ensemble de test: {final_loss:.4f}")
print(f"Perplexité finale: {final_perplexity:.4f}")

# Create a baseline model for comparison
def baseline_perplexity(text, tokenizer):
    # Count frequency of each character
    char_counts = {}
    for ch in text:
        char_counts[ch] = char_counts.get(ch, 0) + 1
    
    # Find most common character
    total_chars = len(text)
    char_probs = {ch: count/total_chars for ch, count in char_counts.items()}
    
    # Calculate negative log likelihood
    total_nll = 0
    for ch in test_text:
        prob = char_probs.get(ch, 1e-10)  # Avoid log(0)
        total_nll -= math.log(prob)
    
    # Calculate perplexity
    avg_nll = total_nll / len(test_text)
    baseline_perp = math.exp(avg_nll)
    
    return baseline_perp

# Calculate baseline perplexity
baseline_perp = baseline_perplexity(train_text, tokenizer)
print(f"Perplexité du modèle de base: {baseline_perp:.4f}")
print(f"Amélioration par rapport au modèle de base: {baseline_perp/final_perplexity:.2f}x")