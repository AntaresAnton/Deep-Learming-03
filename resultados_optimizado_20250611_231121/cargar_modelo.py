# Script para cargar el mejor modelo
import torch
import torch.nn as nn

# Definir la clase del modelo (copiar la clase correspondiente)
class OptimizedLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers=1, dropout=0.2):
        super(OptimizedLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        output = self.dropout(output[:, -1, :])
        return self.fc(output)

# Cargar modelo
checkpoint = torch.load('mejor_modelo.pth', map_location='cpu')
config = checkpoint['config']

# Crear modelo
vocab_size = len(checkpoint['char_to_idx'])
model = OptimizedLSTM(vocab_size, config['embedding_dim'], config['hidden_size'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print("Modelo cargado exitosamente!")
print(f"Accuracy: {checkpoint['results']['accuracy']:.4f}")
