# Proyecto NLP: Análisis de Diálogos con RNN y Transformers
# Optimizado para NVIDIA GTX 1660 Ti
# Aplicando metodología CRISP-DM

# Importar librerías
import os
import gc
import re
import ast
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Configurar semilla para reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Descargar recursos NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Configurar dispositivo (GPU/CPU)
if torch.cuda.is_available():
    try:
        # Probar con un tensor pequeño
        test_tensor = torch.zeros(1, device='cuda')
        device = torch.device('cuda')
        # Configurar para optimizar memoria en GTX 1660 Ti
        torch.cuda.set_per_process_memory_fraction(0.7)  # Usar 70% de la memoria disponible
        print(f"Usando dispositivo: {torch.cuda.get_device_name(0)}")
    except RuntimeError:
        device = torch.device('cpu')
        print("Error al acceder a GPU, usando CPU")
else:
    device = torch.device('cpu')
    print("Usando dispositivo: CPU")

# Función para liberar memoria CUDA
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ====================== FASE 1: COMPRENSIÓN DEL NEGOCIO ======================
print("\n=== FASE 1: COMPRENSIÓN DEL NEGOCIO ===")
print("""
Objetivo del proyecto:
- Desarrollar modelos de NLP para clasificar actos de diálogo y emociones en conversaciones
- Comparar el rendimiento de diferentes arquitecturas (RNN, LSTM, GRU, Transformer)
- Analizar el impacto de hiperparámetros en el rendimiento de los modelos
- Evaluar los modelos con métricas como accuracy, precision, recall y F1-score

Contexto:
- Los actos de diálogo representan la intención comunicativa (pregunta, afirmación, etc.)
- Las emociones representan el estado afectivo expresado en el diálogo
- Ambas clasificaciones son importantes para sistemas de diálogo y análisis de conversaciones
""")

# ====================== FASE 2: COMPRENSIÓN DE LOS DATOS ======================
print("\n=== FASE 2: COMPRENSIÓN DE LOS DATOS ===")

def load_data(train_path='train.csv', val_path='validation.csv', test_path='test.csv'):
    """Carga los datos de entrenamiento, validación y prueba"""
    print("Cargando datos...")
    
    # Función para procesar el formato de los datos
    def process_dataframe(df):
        # Convertir columnas de texto a formato adecuado
        df['dialog'] = df['dialog'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        
        # Convertir columnas numéricas a formato adecuado
        for col in ['act', 'emotion']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else x)
        
        return df
    
    # Cargar datos
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)
    test_data = pd.read_csv(test_path)
    
    # Procesar dataframes
    train_data = process_dataframe(train_data)
    val_data = process_dataframe(val_data)
    test_data = process_dataframe(test_data)
    
    return train_data, val_data, test_data

# Cargar datos
try:
    train_data, val_data, test_data = load_data()
    print("Datos cargados correctamente.")
except Exception as e:
    print(f"Error al cargar datos: {e}")
    # Crear estructura básica si hay error
    train_data = pd.DataFrame(columns=['dialog', 'act', 'emotion'])
    val_data = pd.DataFrame(columns=['dialog', 'act', 'emotion'])
    test_data = pd.DataFrame(columns=['dialog', 'act', 'emotion'])

# Explorar estructura de los datos
print("\nEstructura de los datos:")
print(f"Tamaño del conjunto de entrenamiento: {len(train_data)}")
print(f"Tamaño del conjunto de validación: {len(val_data)}")
print(f"Tamaño del conjunto de prueba: {len(test_data)}")
print(f"Columnas en los datos: {list(train_data.columns)}")

# Expandir los datos para tener una fila por cada diálogo
def expand_dialogues(df):
    """Expande el dataframe para tener una fila por cada diálogo"""
    expanded_data = []
    
    for _, row in df.iterrows():
        dialogues = row['dialog']
        acts = row['act']
        emotions = row['emotion']
        
        # Verificar que las longitudes coincidan
        if len(dialogues) != len(acts) or len(dialogues) != len(emotions):
            print(f"Advertencia: Longitudes inconsistentes - diálogos: {len(dialogues)}, actos: {len(acts)}, emociones: {len(emotions)}")
            # Ajustar longitudes si es necesario
            min_len = min(len(dialogues), len(acts), len(emotions))
            dialogues = dialogues[:min_len]
            acts = acts[:min_len]
            emotions = emotions[:min_len]
        
        # Expandir cada diálogo
        for i in range(len(dialogues)):
            expanded_data.append({
                'dialog': dialogues[i],
                'act': acts[i],
                'emotion': emotions[i]
            })
    
    return pd.DataFrame(expanded_data)

# Expandir datos
try:
    train_expanded = expand_dialogues(train_data)
    val_expanded = expand_dialogues(val_data)
    test_expanded = expand_dialogues(test_data)
    
    print("\nDatos expandidos:")
    print(f"Tamaño del conjunto de entrenamiento expandido: {len(train_expanded)}")
    print(f"Tamaño del conjunto de validación expandido: {len(val_expanded)}")
    print(f"Tamaño del conjunto de prueba expandido: {len(test_expanded)}")
except Exception as e:
    print(f"Error al expandir datos: {e}")
    # Crear datos expandidos básicos si hay error
    train_expanded = pd.DataFrame(columns=['dialog', 'act', 'emotion'])
    val_expanded = pd.DataFrame(columns=['dialog', 'act', 'emotion'])
    test_expanded = pd.DataFrame(columns=['dialog', 'act', 'emotion'])

# Mapear códigos numéricos a etiquetas significativas
def map_labels(df):
    """Mapea códigos numéricos a etiquetas significativas"""
    # Mapeo de actos de diálogo (basado en la documentación o inferencia)
    act_mapping = {
        0: 'statement',
        1: 'inform',
        2: 'question',
        3: 'directive',
        4: 'commissive'
    }
    
    # Mapeo de emociones (basado en la documentación o inferencia)
    emotion_mapping = {
        0: 'neutral',
        1: 'anger',
        2: 'disgust',
        3: 'fear',
        4: 'happiness',
        5: 'sadness',
        6: 'surprise'
    }
    
    # Aplicar mapeo
    df['act_label'] = df['act'].map(act_mapping)
    df['emotion_label'] = df['emotion'].map(emotion_mapping)
    
    return df

# Mapear etiquetas
train_expanded = map_labels(train_expanded)
val_expanded = map_labels(val_expanded)
test_expanded = map_labels(test_expanded)

# Análisis exploratorio de datos
print("\n--- ANÁLISIS EXPLORATORIO DE DATOS ---")

# Distribución de actos de diálogo
print("Distribución de actos de diálogo (train):")
act_counts = train_expanded['act_label'].value_counts()
print(act_counts)

# Distribución de emociones
print("Distribución de emociones (train):")
emotion_counts = train_expanded['emotion_label'].value_counts()
print(emotion_counts)

# Visualizar distribuciones
plt.figure(figsize=(15, 6))

# Gráfico de distribución de actos
plt.subplot(1, 2, 1)
sns.countplot(y='act_label', data=train_expanded, palette='viridis')
plt.title('Distribución de Actos de Diálogo', fontsize=14)
plt.xlabel('Frecuencia', fontsize=12)
plt.ylabel('Acto', fontsize=12)

# Gráfico de distribución de emociones
plt.subplot(1, 2, 2)
sns.countplot(y='emotion_label', data=train_expanded, palette='viridis')
plt.title('Distribución de Emociones', fontsize=14)
plt.xlabel('Frecuencia', fontsize=12)
plt.ylabel('Emoción', fontsize=12)

plt.tight_layout()
plt.savefig('distribucion_clases.png')
plt.close()

# Estadísticas de longitud de diálogos
train_expanded['dialog_length'] = train_expanded['dialog'].apply(len)
print("\nEstadísticas de longitud de diálogos (caracteres):")
print(f"Media: {train_expanded['dialog_length'].mean():.2f}")
print(f"Mediana: {train_expanded['dialog_length'].median():.2f}")
print(f"Máximo: {train_expanded['dialog_length'].max()}")
print(f"Mínimo: {train_expanded['dialog_length'].min()}")

# Visualizar distribución de longitudes
plt.figure(figsize=(10, 6))
sns.histplot(train_expanded['dialog_length'], bins=30, kde=True)
plt.title('Distribución de Longitud de Diálogos', fontsize=14)
plt.xlabel('Número de Caracteres', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.savefig('distribucion_longitudes_caracteres.png')
plt.close()

# Relación entre actos y emociones
plt.figure(figsize=(12, 8))
cross_tab = pd.crosstab(train_expanded['act_label'], train_expanded['emotion_label'])
sns.heatmap(cross_tab, annot=True, cmap='YlGnBu', fmt='d')
plt.title('Relación entre Actos de Diálogo y Emociones', fontsize=14)
plt.xlabel('Emoción', fontsize=12)
plt.ylabel('Acto de Diálogo', fontsize=12)
plt.savefig('relacion_actos_emociones.png')
plt.close()

# ====================== FASE 3: PREPARACIÓN DE LOS DATOS ======================
print("\n=== FASE 3: PREPARACIÓN DE LOS DATOS ===")

# Preprocesamiento de texto
print("\n--- PREPROCESAMIENTO DE TEXTO ---")

def preprocess_text(text):
    """Preprocesa texto para análisis"""
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales y números
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Aplicar preprocesamiento
print("Aplicando preprocesamiento de texto...")
train_expanded['processed_text'] = train_expanded['dialog'].apply(preprocess_text)
val_expanded['processed_text'] = val_expanded['dialog'].apply(preprocess_text)
test_expanded['processed_text'] = test_expanded['dialog'].apply(preprocess_text)

# Tokenización
print("Tokenizando textos...")
train_expanded['tokens'] = train_expanded['processed_text'].apply(word_tokenize)
val_expanded['tokens'] = val_expanded['processed_text'].apply(word_tokenize)
test_expanded['tokens'] = test_expanded['processed_text'].apply(word_tokenize)

# Estadísticas de longitud de tokens
token_lengths = train_expanded['tokens'].apply(len)
print("Estadísticas de longitud de texto (palabras):")
print(f"Media: {token_lengths.mean():.2f}")
print(f"Mediana: {token_lengths.median():.2f}")
print(f"Máximo: {token_lengths.max()}")
print(f"Mínimo: {token_lengths.min()}")

# Determinar longitud máxima para secuencias
max_len = int(np.percentile(token_lengths, 95))
print(f"Longitud máxima de secuencia (percentil 95): {max_len}")

# Visualizar distribución de longitudes de tokens
plt.figure(figsize=(10, 6))
sns.histplot(token_lengths, bins=20, kde=True)
plt.axvline(x=max_len, color='r', linestyle='--', label=f'Max Length ({max_len})')
plt.title('Distribución de Longitud de Textos', fontsize=14)
plt.xlabel('Número de Tokens', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.legend()
plt.savefig('distribucion_longitudes_tokens.png')
plt.close()

# Codificación de etiquetas
print("\n--- CODIFICACIÓN DE ETIQUETAS ---")

# Mapeo de actos de diálogo a índices
act_to_idx = {act: idx for idx, act in enumerate(sorted(train_expanded['act'].unique()))}
idx_to_act = {idx: act for act, idx in act_to_idx.items()}
print(f"Número de clases únicas para 'act': {len(act_to_idx)}")
print(f"Mapeo de actos: {act_to_idx}")

# Mapeo de emociones a índices
emotion_to_idx = {emotion: idx for idx, emotion in enumerate(sorted(train_expanded['emotion'].unique()))}
idx_to_emotion = {idx: emotion for emotion, idx in emotion_to_idx.items()}
print(f"Número de clases únicas para 'emotion': {len(emotion_to_idx)}")
print(f"Mapeo de emociones: {emotion_to_idx}")

# Construcción del vocabulario
print("\n--- CONSTRUCCIÓN DEL VOCABULARIO ---")

# Contar frecuencia de palabras
word_counts = Counter()
for tokens in train_expanded['tokens']:
    word_counts.update(tokens)

# Crear vocabulario
# Añadir tokens especiales: <pad> para padding, <unk> para palabras desconocidas
word_to_idx = {'<pad>': 0, '<unk>': 1}
for word, _ in word_counts.most_common():
    if word not in word_to_idx:
        word_to_idx[word] = len(word_to_idx)

print(f"Tamaño del vocabulario: {len(word_to_idx)}")
print("Palabras más comunes:")
for word, count in word_counts.most_common(20):
    print(f"{word}: {count}")

# Visualizar distribución de frecuencias de palabras
plt.figure(figsize=(12, 6))
word_freq = pd.DataFrame(word_counts.most_common(30), columns=['word', 'frequency'])
sns.barplot(x='word', y='frequency', data=word_freq)
plt.title('Frecuencia de las 30 Palabras Más Comunes', fontsize=14)
plt.xlabel('Palabra', fontsize=12)
plt.ylabel('Frecuencia', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('frecuencia_palabras.png')
plt.close()

# Función para convertir texto a secuencia de índices
def text_to_sequence(text, word_to_idx, max_len):
    """Convierte texto a secuencia de índices con padding"""
    if isinstance(text, str):
        tokens = word_tokenize(preprocess_text(text))
    else:
        tokens = text
    
    # Truncar si es necesario
    tokens = tokens[:max_len]
    
    # Convertir tokens a índices
    sequence = [word_to_idx.get(token, word_to_idx['<unk>']) for token in tokens]
    
    # Añadir padding si es necesario
    if len(sequence) < max_len:
        sequence += [word_to_idx['<pad>']] * (max_len - len(sequence))
    
    return sequence

# Dataset personalizado
class DialogDataset(Dataset):
    """Dataset para diálogos con actos y emociones"""
    def __init__(self, data, word_to_idx, act_to_idx, emotion_to_idx, max_len, task='both'):
        self.data = data
        self.word_to_idx = word_to_idx
        self.act_to_idx = act_to_idx
        self.emotion_to_idx = emotion_to_idx
        self.max_len = max_len
        self.task = task  # 'act', 'emotion', o 'both'
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data.iloc[idx]['tokens']
        act = self.data.iloc[idx]['act']
        emotion = self.data.iloc[idx]['emotion']
        
        # Convertir a secuencia de índices
        text_seq = text_to_sequence(text, self.word_to_idx, self.max_len)
        
        # Convertir a tensores
        text_tensor = torch.tensor(text_seq, dtype=torch.long)
        act_tensor = torch.tensor(self.act_to_idx[act], dtype=torch.long)
        emotion_tensor = torch.tensor(self.emotion_to_idx[emotion], dtype=torch.long)
        
        if self.task == 'act':
            return {'text': text_tensor, 'label': act_tensor}
        elif self.task == 'emotion':
            return {'text': text_tensor, 'label': emotion_tensor}
        else:  # 'both'
            return {
                'text': text_tensor,
                'act': act_tensor,
                'emotion': emotion_tensor
            }

# Crear datasets
print("Creando datasets...")
# Reducir batch_size para evitar problemas de memoria en GTX 1660 Ti
batch_size = 16

# Datasets para clasificación de actos
train_act_dataset = DialogDataset(train_expanded, word_to_idx, act_to_idx, emotion_to_idx, max_len, task='act')
val_act_dataset = DialogDataset(val_expanded, word_to_idx, act_to_idx, emotion_to_idx, max_len, task='act')
test_act_dataset = DialogDataset(test_expanded, word_to_idx, act_to_idx, emotion_to_idx, max_len, task='act')

# Datasets para clasificación de emociones
train_emotion_dataset = DialogDataset(train_expanded, word_to_idx, act_to_idx, emotion_to_idx, max_len, task='emotion')
val_emotion_dataset = DialogDataset(val_expanded, word_to_idx, act_to_idx, emotion_to_idx, max_len, task='emotion')
test_emotion_dataset = DialogDataset(test_expanded, word_to_idx, act_to_idx, emotion_to_idx, max_len, task='emotion')

# DataLoaders para actos
train_act_loader = DataLoader(train_act_dataset, batch_size=batch_size, shuffle=True)
val_act_loader = DataLoader(val_act_dataset, batch_size=batch_size)
test_act_loader = DataLoader(test_act_dataset, batch_size=batch_size)

# DataLoaders para emociones
train_emotion_loader = DataLoader(train_emotion_dataset, batch_size=batch_size, shuffle=True)
val_emotion_loader = DataLoader(val_emotion_dataset, batch_size=batch_size)
test_emotion_loader = DataLoader(test_emotion_dataset, batch_size=batch_size)

print(f"Número de batches en train_act_loader: {len(train_act_loader)}")
print(f"Número de batches en val_act_loader: {len(val_act_loader)}")
print(f"Número de batches en test_act_loader: {len(test_act_loader)}")

# ====================== FASE 4: MODELADO ======================
print("\n=== FASE 4: MODELADO ===")

# Definición de modelos
print("\n--- DEFINICIÓN DE MODELOS ---")

# Parámetros reducidos para evitar problemas de memoria en GTX 1660 Ti
embedding_dim = 64
hidden_dim = 128
n_layers = 2
dropout = 0.3

# Modelo RNN simple
class SimpleRNN(nn.Module):
    """Modelo RNN simple para clasificación de texto"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5):
        super(SimpleRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, hidden = self.rnn(embedded)
        # output shape: [batch_size, seq_len, hidden_dim]
        # hidden shape: [n_layers, batch_size, hidden_dim]
        
        # Usar el último estado oculto para la clasificación
        hidden = hidden[-1, :, :]
        # hidden shape: [batch_size, hidden_dim]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# Modelo LSTM
class LSTM(nn.Module):
    """Modelo LSTM para clasificación de texto"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5, bidirectional=False):
        super(LSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, 
                           dropout=dropout if n_layers > 1 else 0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        
        # Si es bidireccional, multiplicamos por 2 la dimensión de salida
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden shape: [n_layers * num_directions, batch_size, hidden_dim]
        
        # Si es bidireccional, concatenamos los estados ocultos finales
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        # hidden shape: [batch_size, hidden_dim * num_directions]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# Modelo GRU
class GRU(nn.Module):
    """Modelo GRU para clasificación de texto"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers=1, dropout=0.5, bidirectional=False):
        super(GRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers=n_layers, batch_first=True, 
                         dropout=dropout if n_layers > 1 else 0, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        
        # Si es bidireccional, multiplicamos por 2 la dimensión de salida
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc = nn.Linear(fc_input_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        output, hidden = self.gru(embedded)
        # output shape: [batch_size, seq_len, hidden_dim * num_directions]
        # hidden shape: [n_layers * num_directions, batch_size, hidden_dim]
        
        # Si es bidireccional, concatenamos los estados ocultos finales
        if self.gru.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
        # hidden shape: [batch_size, hidden_dim * num_directions]
        
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# Modelo Transformer
class TransformerEncoder(nn.Module):
    """Modelo Transformer para clasificación de texto"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_heads=4, n_layers=2, dropout=0.5):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # Codificación posicional
        self.pos_encoder = nn.Dropout(dropout)
        
        # Capas de encoder del transformer
        encoder_layers = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=n_heads, 
                                                   dim_feedforward=hidden_dim, dropout=dropout,
                                                   batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=n_layers)
        
        # Capa de clasificación
        self.fc = nn.Linear(embedding_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch_size, seq_len]
        
        # Crear máscara para padding
        mask = (text == 0)
        
        embedded = self.embedding(text)
        # embedded shape: [batch_size, seq_len, embedding_dim]
        
        embedded = self.pos_encoder(embedded)
        
        # Aplicar transformer
        encoded = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        # encoded shape: [batch_size, seq_len, embedding_dim]
        
        # Promedio global para obtener representación de secuencia
        # Ignorar tokens de padding en el promedio
        mask_expanded = mask.unsqueeze(-1).expand(encoded.size())
        encoded = encoded.masked_fill(mask_expanded, 0)
        
        # Sumar a lo largo de la dimensión de secuencia y dividir por la longitud real
        seq_lengths = (~mask).sum(dim=1).unsqueeze(-1)
        seq_lengths = torch.clamp(seq_lengths, min=1)  # Evitar división por cero
        pooled = encoded.sum(dim=1) / seq_lengths
        
        pooled = self.dropout(pooled)
        return self.fc(pooled)

# Visualización de arquitecturas
def visualize_model_architecture(model_name):
    """Visualiza la arquitectura del modelo de forma gráfica"""
    plt.figure(figsize=(12, 8))
    
    if model_name == 'SimpleRNN':
        # Visualización de RNN
        layers = ['Embedding\n(vocab_size, embedding_dim)', 
                  'RNN\n(embedding_dim, hidden_dim)', 
                  'Dropout', 
                  'Linear\n(hidden_dim, output_dim)']
        
    elif model_name == 'LSTM':
        # Visualización de LSTM
        layers = ['Embedding\n(vocab_size, embedding_dim)', 
                  'LSTM\n(embedding_dim, hidden_dim)', 
                  'Dropout', 
                  'Linear\n(hidden_dim, output_dim)']
        
    elif model_name == 'GRU':
        # Visualización de GRU
        layers = ['Embedding\n(vocab_size, embedding_dim)', 
                  'GRU\n(embedding_dim, hidden_dim)', 
                  'Dropout', 
                  'Linear\n(hidden_dim, output_dim)']
        
    elif model_name == 'Transformer':
        # Visualización de Transformer
        layers = ['Embedding\n(vocab_size, embedding_dim)', 
                  'Positional Encoding', 
                  'TransformerEncoder\n(n_layers, n_heads)', 
                  'Global Pooling', 
                  'Dropout', 
                  'Linear\n(embedding_dim, output_dim)']
    
    # Dibujar capas
    for i, layer in enumerate(layers):
        plt.text(0.5, 1 - (i+1)/(len(layers)+1), layer, 
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox=dict(facecolor='lightblue', alpha=0.5, boxstyle='round,pad=0.5'))
        
        # Dibujar flechas entre capas
        if i < len(layers) - 1:
            plt.arrow(0.5, 1 - (i+1)/(len(layers)+1) - 0.05, 
                      0, -1/(len(layers)+1) + 0.1, 
                      head_width=0.05, head_length=0.02, fc='black', ec='black')
    
    plt.axis('off')
    plt.title(f'Arquitectura del modelo {model_name}', fontsize=16)
    plt.savefig(f'arquitectura_{model_name}.png')
    plt.close()

# Visualizar arquitecturas
for model_name in ['SimpleRNN', 'LSTM', 'GRU', 'Transformer']:
    visualize_model_architecture(model_name)

# Función para entrenar modelo
def train_model(model, train_loader, val_loader, optimizer, criterion, n_epochs, model_name, task_name, patience=3):
    """Entrena el modelo y devuelve historial de métricas"""
    # Mover modelo al dispositivo
    model = model.to(device)
    
    # Historial de métricas
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Early stopping
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(n_epochs):
        # Modo entrenamiento
        model.train()
        epoch_loss = 0
        epoch_acc = 0
        
        # Barra de progreso
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
        
        for batch in progress_bar:
            # Obtener datos del batch
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Reiniciar gradientes
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(text)
            
            # Calcular pérdida
            loss = criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            
            # Actualizar pesos
            optimizer.step()
            
            # Calcular accuracy
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == labels).float().sum()
            acc = correct / len(labels)
            
            # Actualizar métricas
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
            # Actualizar barra de progreso
            progress_bar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
        
        # Calcular métricas promedio
        train_loss = epoch_loss / len(train_loader)
        train_acc = epoch_acc / len(train_loader)
        
        # Modo evaluación
        model.eval()
        val_loss = 0
        val_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                # Obtener datos del batch
                text = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                # Forward pass
                predictions = model(text)
                
                # Calcular pérdida
                loss = criterion(predictions, labels)
                
                # Calcular accuracy
                _, predicted = torch.max(predictions, 1)
                correct = (predicted == labels).float().sum()
                acc = correct / len(labels)
                
                # Actualizar métricas
                val_loss += loss.item()
                val_acc += acc.item()
        
        # Calcular métricas promedio
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        
        # Actualizar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Imprimir métricas
        print(f'Epoch {epoch+1}/{n_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Guardar mejor modelo
            torch.save(model.state_dict(), f'best_{model_name}_{task_name}.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping after {epoch+1} epochs')
                break
        
        # Liberar memoria
        clear_gpu_memory()
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load(f'best_{model_name}_{task_name}.pt'))
    
    return model, history

# Función para evaluar modelo
def evaluate_model(model, test_loader, criterion, idx_to_label):
    """Evalúa el modelo en el conjunto de prueba"""
    # Mover modelo al dispositivo
    model = model.to(device)
    
    # Modo evaluación
    model.eval()
    
    # Métricas
    test_loss = 0
    test_acc = 0
    
    # Predicciones y etiquetas para métricas detalladas
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Evaluando'):
            # Obtener datos del batch
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            # Forward pass
            predictions = model(text)
            
            # Calcular pérdida
            loss = criterion(predictions, labels)
            
            # Calcular accuracy
            _, predicted = torch.max(predictions, 1)
            correct = (predicted == labels).float().sum()
            acc = correct / len(labels)
            
            # Actualizar métricas
            test_loss += loss.item()
            test_acc += acc.item()
            
            # Guardar predicciones y etiquetas
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calcular métricas promedio
    test_loss = test_loss / len(test_loader)
    test_acc = test_acc / len(test_loader)
    
    # Calcular métricas detalladas
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    # Imprimir métricas
    print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')
    print(f'Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}')
    
    # Matriz de confusión
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Visualizar matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[idx_to_label[i] for i in range(len(idx_to_label))],
                yticklabels=[idx_to_label[i] for i in range(len(idx_to_label))])
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model.__class__.__name__}.png')
    plt.close()
    
    # Reporte de clasificación
    class_report = classification_report(all_labels, all_predictions, 
                                        target_names=[idx_to_label[i] for i in range(len(idx_to_label))],
                                        digits=4)
    print("Reporte de Clasificación:")
    print(class_report)
    
    # Guardar reporte en archivo
    with open(f'classification_report_{model.__class__.__name__}.txt', 'w') as f:
        f.write(class_report)
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'classification_report': class_report
    }

# Función para visualizar historial de entrenamiento
def plot_training_history(history, model_name, task_name):
    """Visualiza el historial de entrenamiento"""
    plt.figure(figsize=(12, 5))
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Gráfico de accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_history_{model_name}_{task_name}.png')
    plt.close()

# Función para comparar modelos
def compare_models(results, task_name):
    """Compara los resultados de diferentes modelos"""
    # Extraer métricas
    models = list(results.keys())
    accuracy = [results[model]['accuracy'] for model in models]
    precision = [results[model]['precision'] for model in models]
    recall = [results[model]['recall'] for model in models]
    f1 = [results[model]['f1'] for model in models]
    
    # Crear dataframe
    df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    })
    
    # Visualizar comparación
    plt.figure(figsize=(12, 8))
    
    # Gráfico de barras para cada métrica
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
    for i, metric in enumerate(metrics):
        plt.subplot(2, 2, i+1)
        sns.barplot(x='Model', y=metric, data=df)
        plt.title(metric)
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{task_name}.png')
    plt.close()
    
    # Imprimir tabla de comparación
    print("Comparación de Modelos:")
    print(df.to_string(index=False))
    
    # Guardar tabla en archivo
    df.to_csv(f'model_comparison_{task_name}.csv', index=False)
    
    return df

# Entrenar y evaluar modelos para clasificación de actos
print("\n--- ENTRENAMIENTO Y EVALUACIÓN DE MODELOS PARA ACTOS DE DIÁLOGO ---")

# Parámetros de entrenamiento
n_epochs = 10
learning_rate = 0.001

# Resultados
act_results = {}

# Modelo SimpleRNN
print("\nEntrenando SimpleRNN para clasificación de actos...")
rnn_act = SimpleRNN(len(word_to_idx), embedding_dim, hidden_dim, len(act_to_idx), n_layers, dropout)
optimizer = optim.Adam(rnn_act.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
rnn_act, rnn_act_history = train_model(rnn_act, train_act_loader, val_act_loader, optimizer, criterion, n_epochs, 'SimpleRNN', 'act')
plot_training_history(rnn_act_history, 'SimpleRNN', 'act')
act_results['SimpleRNN'] = evaluate_model(rnn_act, test_act_loader, criterion, idx_to_act)

# Liberar memoria
clear_gpu_memory()

# Modelo LSTM
print("\nEntrenando LSTM para clasificación de actos...")
lstm_act = LSTM(len(word_to_idx), embedding_dim, hidden_dim, len(act_to_idx), n_layers, dropout)
optimizer = optim.Adam(lstm_act.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
lstm_act, lstm_act_history = train_model(lstm_act, train_act_loader, val_act_loader, optimizer, criterion, n_epochs, 'LSTM', 'act')
plot_training_history(lstm_act_history, 'LSTM', 'act')
act_results['LSTM'] = evaluate_model(lstm_act, test_act_loader, criterion, idx_to_act)

# Liberar memoria
clear_gpu_memory()

# Modelo GRU
print("\nEntrenando GRU para clasificación de actos...")
gru_act = GRU(len(word_to_idx), embedding_dim, hidden_dim, len(act_to_idx), n_layers, dropout)
optimizer = optim.Adam(gru_act.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
gru_act, gru_act_history = train_model(gru_act, train_act_loader, val_act_loader, optimizer, criterion, n_epochs, 'GRU', 'act')
plot_training_history(gru_act_history, 'GRU', 'act')
act_results['GRU'] = evaluate_model(gru_act, test_act_loader, criterion, idx_to_act)

# Liberar memoria
clear_gpu_memory()

# Modelo Transformer
print("\nEntrenando Transformer para clasificación de actos...")
transformer_act = TransformerEncoder(len(word_to_idx), embedding_dim, hidden_dim*2, len(act_to_idx), n_heads=4, n_layers=2, dropout=dropout)
optimizer = optim.Adam(transformer_act.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
transformer_act, transformer_act_history = train_model(transformer_act, train_act_loader, val_act_loader, optimizer, criterion, n_epochs, 'Transformer', 'act')
plot_training_history(transformer_act_history, 'Transformer', 'act')
act_results['Transformer'] = evaluate_model(transformer_act, test_act_loader, criterion, idx_to_act)

# Liberar memoria
clear_gpu_memory()

# Comparar modelos para actos
compare_models(act_results, 'act')

# Entrenar y evaluar modelos para clasificación de emociones
print("\n--- ENTRENAMIENTO Y EVALUACIÓN DE MODELOS PARA EMOCIONES ---")

# Resultados
emotion_results = {}

# Modelo SimpleRNN
print("\nEntrenando SimpleRNN para clasificación de emociones...")
rnn_emotion = SimpleRNN(len(word_to_idx), embedding_dim, hidden_dim, len(emotion_to_idx), n_layers, dropout)
optimizer = optim.Adam(rnn_emotion.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
rnn_emotion, rnn_emotion_history = train_model(rnn_emotion, train_emotion_loader, val_emotion_loader, optimizer, criterion, n_epochs, 'SimpleRNN', 'emotion')
plot_training_history(rnn_emotion_history, 'SimpleRNN', 'emotion')
emotion_results['SimpleRNN'] = evaluate_model(rnn_emotion, test_emotion_loader, criterion, idx_to_emotion)

# Liberar memoria
clear_gpu_memory()

# Modelo LSTM
print("\nEntrenando LSTM para clasificación de emociones...")
lstm_emotion = LSTM(len(word_to_idx), embedding_dim, hidden_dim, len(emotion_to_idx), n_layers, dropout)
optimizer = optim.Adam(lstm_emotion.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
lstm_emotion, lstm_emotion_history = train_model(lstm_emotion, train_emotion_loader, val_emotion_loader, optimizer, criterion, n_epochs, 'LSTM', 'emotion')
plot_training_history(lstm_emotion_history, 'LSTM', 'emotion')
emotion_results['LSTM'] = evaluate_model(lstm_emotion, test_emotion_loader, criterion, idx_to_emotion)

# Liberar memoria
clear_gpu_memory()

# Modelo GRU
print("\nEntrenando GRU para clasificación de emociones...")
gru_emotion = GRU(len(word_to_idx), embedding_dim, hidden_dim, len(emotion_to_idx), n_layers, dropout)
optimizer = optim.Adam(gru_emotion.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
gru_emotion, gru_emotion_history = train_model(gru_emotion, train_emotion_loader, val_emotion_loader, optimizer, criterion, n_epochs, 'GRU', 'emotion')
plot_training_history(gru_emotion_history, 'GRU', 'emotion')
emotion_results['GRU'] = evaluate_model(gru_emotion, test_emotion_loader, criterion, idx_to_emotion)

# Liberar memoria
clear_gpu_memory()

# Modelo Transformer
print("\nEntrenando Transformer para clasificación de emociones...")
transformer_emotion = TransformerEncoder(len(word_to_idx), embedding_dim, hidden_dim*2, len(emotion_to_idx), n_heads=4, n_layers=2, dropout=dropout)
optimizer = optim.Adam(transformer_emotion.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
transformer_emotion, transformer_emotion_history = train_model(transformer_emotion, train_emotion_loader, val_emotion_loader, optimizer, criterion, n_epochs, 'Transformer', 'emotion')
plot_training_history(transformer_emotion_history, 'Transformer', 'emotion')
emotion_results['Transformer'] = evaluate_model(transformer_emotion, test_emotion_loader, criterion, idx_to_emotion)

# Liberar memoria
clear_gpu_memory()

# Comparar modelos para emociones
compare_models(emotion_results, 'emotion')

# ====================== FASE 5: EVALUACIÓN ======================
print("\n=== FASE 5: EVALUACIÓN ===")

# Análisis de hiperparámetros
print("\n--- ANÁLISIS DE HIPERPARÁMETROS ---")

# Función para analizar hiperparámetros
def hyperparameter_analysis(model_class, param_name, param_values, task='act'):
    """Analiza el impacto de un hiperparámetro en el rendimiento del modelo"""
    # Seleccionar dataset según tarea
    if task == 'act':
        train_loader = train_act_loader
        val_loader = val_act_loader
        output_dim = len(act_to_idx)
        idx_to_label = idx_to_act
    else:  # 'emotion'
        train_loader = train_emotion_loader
        val_loader = val_emotion_loader
        output_dim = len(emotion_to_idx)
        idx_to_label = idx_to_emotion
    
    # Resultados
    results = {
        'param_value': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    # Entrenar modelo con diferentes valores del hiperparámetro
    for value in param_values:
        print(f"\nEntrenando {model_class.__name__} con {param_name}={value}...")
        
        # Crear modelo con el valor del hiperparámetro
        if param_name == 'hidden_dim':
            model = model_class(len(word_to_idx), embedding_dim, value, output_dim, n_layers, dropout)
        elif param_name == 'n_layers':
            model = model_class(len(word_to_idx), embedding_dim, hidden_dim, output_dim, value, dropout)
        elif param_name == 'dropout':
            model = model_class(len(word_to_idx), embedding_dim, hidden_dim, output_dim, n_layers, value)
        
        # Entrenar modelo
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        model, history = train_model(model, train_loader, val_loader, optimizer, criterion, 5, f"{model_class.__name__}_{param_name}_{value}", task)
        
        # Evaluar modelo en validación
        model.eval()
        val_loss = 0
        val_acc = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                text = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                predictions = model(text)
                loss = criterion(predictions, labels)
                
                _, predicted = torch.max(predictions, 1)
                correct = (predicted == labels).float().sum()
                acc = correct / len(labels)
                
                val_loss += loss.item()
                val_acc += acc.item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        val_loss = val_loss / len(val_loader)
        val_acc = val_acc / len(val_loader)
        val_f1 = f1_score(all_labels, all_predictions, average='weighted')
        
        # Guardar resultados
        results['param_value'].append(value)
        results['val_loss'].append(val_loss)
        results['val_acc'].append(val_acc)
        results['val_f1'].append(val_f1)
        
        # Liberar memoria
        clear_gpu_memory()
    
    # Visualizar resultados
    plt.figure(figsize=(15, 5))
    
    # Gráfico de pérdida
    plt.subplot(1, 3, 1)
    plt.plot(results['param_value'], results['val_loss'], marker='o')
    plt.title(f'Validation Loss vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Loss')
    
    # Gráfico de accuracy
    plt.subplot(1, 3, 2)
    plt.plot(results['param_value'], results['val_acc'], marker='o')
    plt.title(f'Validation Accuracy vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('Accuracy')
    
    # Gráfico de F1
    plt.subplot(1, 3, 3)
    plt.plot(results['param_value'], results['val_f1'], marker='o')
    plt.title(f'Validation F1 vs {param_name}')
    plt.xlabel(param_name)
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig(f'hyperparameter_{model_class.__name__}_{param_name}_{task}.png')
    plt.close()
    
    return results

# Analizar hiperparámetros para el mejor modelo de actos
best_act_model = max(act_results.items(), key=lambda x: x[1]['f1'])[0]
print(f"\nAnalizando hiperparámetros para el mejor modelo de actos: {best_act_model}")

if best_act_model == 'SimpleRNN':
    model_class = SimpleRNN
elif best_act_model == 'LSTM':
    model_class = LSTM
elif best_act_model == 'GRU':
    model_class = GRU
else:  # 'Transformer'
    model_class = TransformerEncoder

# Analizar hidden_dim
hidden_dim_values = [64, 128, 256]
hidden_dim_results = hyperparameter_analysis(model_class, 'hidden_dim', hidden_dim_values, 'act')

# Analizar n_layers
n_layers_values = [1, 2, 3]
n_layers_results = hyperparameter_analysis(model_class, 'n_layers', n_layers_values, 'act')

# Analizar dropout
dropout_values = [0.1, 0.3, 0.5]
dropout_results = hyperparameter_analysis(model_class, 'dropout', dropout_values, 'act')

# Analizar hiperparámetros para el mejor modelo de emociones
best_emotion_model = max(emotion_results.items(), key=lambda x: x[1]['f1'])[0]
print(f"\nAnalizando hiperparámetros para el mejor modelo de emociones: {best_emotion_model}")

if best_emotion_model == 'SimpleRNN':
    model_class = SimpleRNN
elif best_emotion_model == 'LSTM':
    model_class = LSTM
elif best_emotion_model == 'GRU':
    model_class = GRU
else:  # 'Transformer'
    model_class = TransformerEncoder

# Analizar hidden_dim
hidden_dim_values = [64, 128, 256]
hidden_dim_results = hyperparameter_analysis(model_class, 'hidden_dim', hidden_dim_values, 'emotion')

# Analizar n_layers
n_layers_values = [1, 2, 3]
n_layers_results = hyperparameter_analysis(model_class, 'n_layers', n_layers_values, 'emotion')

# Analizar dropout
dropout_values = [0.1, 0.3, 0.5]
dropout_results = hyperparameter_analysis(model_class, 'dropout', dropout_values, 'emotion')

# ====================== FASE 6: IMPLEMENTACIÓN ======================
print("\n=== FASE 6: IMPLEMENTACIÓN ===")

# Función para predecir con el mejor modelo
def predict_dialog(text, model, word_to_idx, idx_to_label, max_len):
    """Predice la etiqueta para un texto de diálogo"""
    # Preprocesar texto
    processed_text = preprocess_text(text)
    tokens = word_tokenize(processed_text)
    
    # Convertir a secuencia de índices
    sequence = text_to_sequence(tokens, word_to_idx, max_len)
    
    # Convertir a tensor
    text_tensor = torch.tensor(sequence, dtype=torch.long).unsqueeze(0).to(device)
    
    # Predecir
    model.eval()
    with torch.no_grad():
        predictions = model(text_tensor)
        _, predicted = torch.max(predictions, 1)
        label_idx = predicted.item()
        label = idx_to_label[label_idx]
    
    return label

# Cargar los mejores modelos
best_act_model_name = max(act_results.items(), key=lambda x: x[1]['f1'])[0]
best_emotion_model_name = max(emotion_results.items(), key=lambda x: x[1]['f1'])[0]

print(f"Mejor modelo para actos: {best_act_model_name}")
print(f"Mejor modelo para emociones: {best_emotion_model_name}")

# Crear función para demostración
def demo_prediction(text):
    """Demuestra la predicción de actos y emociones para un texto"""
    print(f"\nTexto: {text}")
    
    # Predecir acto
    act = predict_dialog(text, eval(f"{best_act_model_name.lower()}_act"), word_to_idx, idx_to_act, max_len)
    print(f"Acto de diálogo: {act}")
    
    # Predecir emoción
    emotion = predict_dialog(text, eval(f"{best_emotion_model_name.lower()}_emotion"), word_to_idx, idx_to_emotion, max_len)
    print(f"Emoción: {emotion}")

# Ejemplos de demostración
print("\n--- DEMOSTRACIÓN DE PREDICCIONES ---")

demo_texts = [
    "Can you help me with this problem?",
    "I'm so happy to see you again!",
    "Please stop talking and listen to me.",
    "I promise I'll be there on time.",
    "That's absolutely ridiculous, I can't believe it!"
]

for text in demo_texts:
    demo_prediction(text)

# Crear una aplicación simple para predicciones
def create_prediction_app():
    """Crea una aplicación simple para predicciones"""
    print("\n--- APLICACIÓN DE PREDICCIÓN ---")
    print("Ingrese un texto para predecir el acto de diálogo y la emoción (o 'salir' para terminar)")
    
    while True:
        text = input("\nTexto: ")
        if text.lower() == 'salir':
            break
        
        # Predecir acto
        act = predict_dialog(text, eval(f"{best_act_model_name.lower()}_act"), word_to_idx, idx_to_act, max_len)
        
        # Predecir emoción
        emotion = predict_dialog(text, eval(f"{best_emotion_model_name.lower()}_emotion"), word_to_idx, idx_to_emotion, max_len)
        
        print(f"Acto de diálogo: {act}")
        print(f"Emoción: {emotion}")

# Ejecutar aplicación de predicción
create_prediction_app()

# ====================== RESUMEN Y CONCLUSIONES ======================
print("\n=== RESUMEN Y CONCLUSIONES ===")

# Resumen de resultados
print("\n--- RESUMEN DE RESULTADOS ---")

# Crear tabla de resultados para actos
act_summary = pd.DataFrame({
    'Modelo': list(act_results.keys()),
    'Accuracy': [act_results[model]['accuracy'] for model in act_results],
    'Precision': [act_results[model]['precision'] for model in act_results],
    'Recall': [act_results[model]['recall'] for model in act_results],
    'F1': [act_results[model]['f1'] for model in act_results]
})

print("\nResultados para clasificación de actos:")
print(act_summary.to_string(index=False))

# Crear tabla de resultados para emociones
emotion_summary = pd.DataFrame({
    'Modelo': list(emotion_results.keys()),
    'Accuracy': [emotion_results[model]['accuracy'] for model in emotion_results],
    'Precision': [emotion_results[model]['precision'] for model in emotion_results],
    'Recall': [emotion_results[model]['recall'] for model in emotion_results],
    'F1': [emotion_results[model]['f1'] for model in emotion_results]
})

print("\nResultados para clasificación de emociones:")
print(emotion_summary.to_string(index=False))

# Conclusiones
print("\n--- CONCLUSIONES ---")

# Mejor modelo para actos
best_act_model_name = max(act_results.items(), key=lambda x: x[1]['f1'])[0]
best_act_f1 = act_results[best_act_model_name]['f1']
print(f"1. El mejor modelo para clasificación de actos de diálogo es {best_act_model_name} con un F1-score de {best_act_f1:.4f}")

# Mejor modelo para emociones
best_emotion_model_name = max(emotion_results.items(), key=lambda x: x[1]['f1'])[0]
best_emotion_f1 = emotion_results[best_emotion_model_name]['f1']
print(f"2. El mejor modelo para clasificación de emociones es {best_emotion_model_name} con un F1-score de {best_emotion_f1:.4f}")

# Comparación entre arquitecturas
print("3. Comparación entre arquitecturas:")
print("   - RNN vs LSTM/GRU: Las arquitecturas LSTM y GRU generalmente superan a la RNN simple debido a su capacidad para manejar dependencias a largo plazo.")
print("   - LSTM vs GRU: GRU tiende a ser más eficiente computacionalmente, mientras que LSTM puede capturar relaciones más complejas.")
print("   - RNN/LSTM/GRU vs Transformer: Los transformers pueden capturar relaciones globales en el texto sin depender de la secuencialidad, lo que los hace efectivos para tareas de NLP.")

# Impacto de hiperparámetros
print("4. Impacto de hiperparámetros:")
print("   - Dimensión oculta: Aumentar la dimensión oculta generalmente mejora el rendimiento hasta cierto punto, después del cual puede causar sobreajuste.")
print("   - Número de capas: Más capas pueden capturar relaciones más complejas, pero también aumentan el riesgo de sobreajuste y problemas de gradientes.")
print("   - Dropout: Un valor óptimo de dropout ayuda a prevenir el sobreajuste, especialmente en modelos más complejos.")

# Desafíos y limitaciones
print("5. Desafíos y limitaciones:")
print("   - Desbalance de clases: Algunas clases están subrepresentadas, lo que afecta el rendimiento del modelo.")
print("   - Tamaño del conjunto de datos: Un conjunto de datos más grande podría mejorar el rendimiento, especialmente para modelos complejos como los transformers.")
print("   - Contexto limitado: Los modelos actuales no consideran el contexto completo de la conversación, solo enunciados individuales.")

# Trabajo futuro
print("6. Trabajo futuro:")
print("   - Incorporar información contextual de la conversación completa.")
print("   - Explorar técnicas de aumento de datos para clases subrepresentadas.")
print("   - Implementar modelos pre-entrenados como BERT o GPT para mejorar el rendimiento.")
print("   - Desarrollar un sistema end-to-end que combine la clasificación de actos y emociones.")

# Guardar conclusiones en archivo
with open('conclusiones.txt', 'w') as f:
    f.write("=== CONCLUSIONES ===\n\n")
    f.write(f"1. El mejor modelo para clasificación de actos de diálogo es {best_act_model_name} con un F1-score de {best_act_f1:.4f}\n\n")
    f.write(f"2. El mejor modelo para clasificación de emociones es {best_emotion_model_name} con un F1-score de {best_emotion_f1:.4f}\n\n")
    f.write("3. Comparación entre arquitecturas:\n")
    f.write("   - RNN vs LSTM/GRU: Las arquitecturas LSTM y GRU generalmente superan a la RNN simple debido a su capacidad para manejar dependencias a largo plazo.\n")
    f.write("   - LSTM vs GRU: GRU tiende a ser más eficiente computacionalmente, mientras que LSTM puede capturar relaciones más complejas.\n")
    f.write("   - RNN/LSTM/GRU vs Transformer: Los transformers pueden capturar relaciones globales en el texto sin depender de la secuencialidad, lo que los hace efectivos para tareas de NLP.\n\n")
    f.write("4. Impacto de hiperparámetros:\n")
    f.write("   - Dimensión oculta: Aumentar la dimensión oculta generalmente mejora el rendimiento hasta cierto punto, después del cual puede causar sobreajuste.\n")
    f.write("   - Número de capas: Más capas pueden capturar relaciones más complejas, pero también aumentan el riesgo de sobreajuste y problemas de gradientes.\n")
    f.write("   - Dropout: Un valor óptimo de dropout ayuda a prevenir el sobreajuste, especialmente en modelos más complejos.\n\n")
    f.write("5. Desafíos y limitaciones:\n")
    f.write("   - Desbalance de clases: Algunas clases están subrepresentadas, lo que afecta el rendimiento del modelo.\n")
    f.write("   - Tamaño del conjunto de datos: Un conjunto de datos más grande podría mejorar el rendimiento, especialmente para modelos complejos como los transformers.\n")
    f.write("   - Contexto limitado: Los modelos actuales no consideran el contexto completo de la conversación, solo enunciados individuales.\n\n")
    f.write("6. Trabajo futuro:\n")
    f.write("   - Incorporar información contextual de la conversación completa.\n")
    f.write("   - Explorar técnicas de aumento de datos para clases subrepresentadas.\n")
    f.write("   - Implementar modelos pre-entrenados como BERT o GPT para mejorar el rendimiento.\n")
    f.write("   - Desarrollar un sistema end-to-end que combine la clasificación de actos y emociones.\n")

print("\n¡Análisis completo! Los resultados y visualizaciones se han guardado en archivos.")
