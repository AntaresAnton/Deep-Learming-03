# Importación de librerías necesarias
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re
import os
import time
import json
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge import Rouge
from tqdm import tqdm
import warnings
from colorama import Fore, Style, init

# Inicializar colorama para colores en la terminal
init(autoreset=True)

# Verificar si CUDA está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{Fore.CYAN}Utilizando dispositivo: {Fore.YELLOW}{device}{Style.RESET_ALL}")

# Configuración de semilla para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Clase para procesar y preparar el dataset de diálogos
class DialogProcessor:
    def __init__(self, train_path, test_path, val_path, max_vocab_size=10000, max_seq_length=50):
        """
        Inicializa el procesador de diálogos
        Args:
            train_path: Ruta al archivo de entrenamiento
            test_path: Ruta al archivo de prueba
            val_path: Ruta al archivo de validación
            max_vocab_size: Tamaño máximo del vocabulario
            max_seq_length: Longitud máxima de secuencia
        """
        print(f"{Fore.GREEN}Inicializando procesador de diálogos...{Style.RESET_ALL}")
        self.train_path = train_path
        self.test_path = test_path
        self.val_path = val_path
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        
        # Cargar datos
        self.train_df = self._load_data(train_path)
        self.test_df = self._load_data(test_path)
        self.val_df = self._load_data(val_path)
        
        # Construir vocabulario
        self.word2idx, self.idx2word = self._build_vocabulary()
        self.vocab_size = len(self.word2idx)
        
        # Mapeo de etiquetas
        self.act_classes = 5  # 0-4 (basado en los ejemplos)
        self.emotion_classes = 7  # 0-6 (basado en los ejemplos)
        
        print(f"{Fore.GREEN}Procesador de diálogos inicializado correctamente.{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Estadísticas:{Style.RESET_ALL}")
        print(f"  - Tamaño del vocabulario: {self.vocab_size}")
        print(f"  - Ejemplos de entrenamiento: {len(self.train_df)}")
        print(f"  - Ejemplos de prueba: {len(self.test_df)}")
        print(f"  - Ejemplos de validación: {len(self.val_df)}")
    
    def _load_data(self, file_path):
        """
        Carga y procesa un archivo CSV de diálogos
        Args:
            file_path: Ruta al archivo CSV
        Returns:
            DataFrame procesado
        """
        print(f"{Fore.BLUE}Cargando datos desde: {file_path}{Style.RESET_ALL}")
        df = pd.read_csv(file_path)
        
        # Convertir strings de listas a listas reales
        df['dialog'] = df['dialog'].apply(self._parse_list)
        df['act'] = df['act'].apply(self._parse_list)
        df['emotion'] = df['emotion'].apply(self._parse_list)
        
        return df
    
    def _parse_list(self, text):
        """
        Convierte una cadena de texto con formato de lista a una lista real
        Args:
            text: Cadena de texto con formato de lista
        Returns:
            Lista procesada
        """
        try:
            return ast.literal_eval(text)
        except:
            return []
    
    def _build_vocabulary(self):
        """
        Construye el vocabulario a partir de los diálogos
        Returns:
            Diccionarios word2idx e idx2word
        """
        print(f"{Fore.BLUE}Construyendo vocabulario...{Style.RESET_ALL}")
        # Tokens especiales
        special_tokens = ['<PAD>', '<UNK>', '<SOS>', '<EOS>']
        
        # Recopilar todas las palabras
        word_freq = {}
        
        # Función para procesar un diálogo y actualizar frecuencias
        def process_dialog(dialog):
            for utterance in dialog:
                # Limpiar y normalizar texto
                utterance = utterance.strip().lower()
                utterance = re.sub(r'[^\w\s\']', ' ', utterance)
                words = utterance.split()
                for word in words:
                    if word in word_freq:
                        word_freq[word] += 1
                    else:
                        word_freq[word] = 1
        
        # Procesar todos los diálogos
        for df in [self.train_df, self.test_df, self.val_df]:
            for dialog in df['dialog']:
                process_dialog(dialog)
        
        # Ordenar por frecuencia y limitar tamaño
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_vocab_size - len(special_tokens)]
        
        # Crear mapeos
        word2idx = {word: idx + len(special_tokens) for idx, (word, _) in enumerate(sorted_words)}
        
        # Añadir tokens especiales
        for idx, token in enumerate(special_tokens):
            word2idx[token] = idx
        
        idx2word = {idx: word for word, idx in word2idx.items()}
        
        print(f"{Fore.GREEN}Vocabulario construido con {len(word2idx)} palabras.{Style.RESET_ALL}")
        return word2idx, idx2word
    
    def tokenize(self, text):
        """
        Convierte un texto a una secuencia de índices
        Args:
            text: Texto a tokenizar
        Returns:
            Lista de índices
        """
        # Limpiar y normalizar texto
        text = text.strip().lower()
        text = re.sub(r'[^\w\s\']', ' ', text)
        words = text.split()
        
        # Convertir a índices
        indices = []
        for word in words:
            if word in self.word2idx:
                indices.append(self.word2idx[word])
            else:
                indices.append(self.word2idx['<UNK>'])
        
        return indices
    
    def prepare_sequence_data(self):
        """
        Prepara los datos para la tarea de generación de secuencias
        Returns:
            Datos de entrenamiento, prueba y validación
        """
        print(f"{Fore.BLUE}Preparando datos para generación de secuencias...{Style.RESET_ALL}")
        
        def create_input_target_pairs(df):
            inputs, targets = [], []
            for dialog in df['dialog']:
                if len(dialog) < 2:  # Necesitamos al menos 2 turnos
                    continue
                for i in range(len(dialog) - 1):
                    input_text = dialog[i]
                    target_text = dialog[i+1]
                    
                    # Tokenizar
                    input_indices = self.tokenize(input_text)
                    target_indices = [self.word2idx['<SOS>']] + self.tokenize(target_text) + [self.word2idx['<EOS>']]
                    
                    # Truncar si es necesario
                    if len(input_indices) > self.max_seq_length:
                        input_indices = input_indices[:self.max_seq_length]
                    if len(target_indices) > self.max_seq_length:
                        target_indices = target_indices[:self.max_seq_length-1] + [self.word2idx['<EOS>']]
                    
                    # Padding
                    input_padded = input_indices + [self.word2idx['<PAD>']] * (self.max_seq_length - len(input_indices))
                    target_padded = target_indices + [self.word2idx['<PAD>']] * (self.max_seq_length - len(target_indices))
                    
                    inputs.append(input_padded)
                    targets.append(target_padded)
            
            return torch.LongTensor(inputs), torch.LongTensor(targets)
        
        train_inputs, train_targets = create_input_target_pairs(self.train_df)
        val_inputs, val_targets = create_input_target_pairs(self.val_df)
        test_inputs, test_targets = create_input_target_pairs(self.test_df)
        
        print(f"{Fore.GREEN}Datos preparados para generación de secuencias.{Style.RESET_ALL}")
        print(f"  - Ejemplos de entrenamiento: {len(train_inputs)}")
        print(f"  - Ejemplos de validación: {len(val_inputs)}")
        print(f"  - Ejemplos de prueba: {len(test_inputs)}")
        
        return {
            'train': (train_inputs, train_targets),
            'val': (val_inputs, val_targets),
            'test': (test_inputs, test_targets)
        }
    
    def prepare_classification_data(self):
        """
        Prepara los datos para las tareas de clasificación (actos y emociones)
        Returns:
            Datos de entrenamiento, prueba y validación
        """
        print(f"{Fore.BLUE}Preparando datos para clasificación...{Style.RESET_ALL}")
        
        def extract_features_and_labels(df):
            features, act_labels, emotion_labels = [], [], []
            
            for i, row in df.iterrows():
                dialog = row['dialog']
                acts = row['act']
                emotions = row['emotion']
                
                # Verificar que las longitudes coincidan
                if len(dialog) != len(acts) or len(dialog) != len(emotions):
                    continue
                
                for j in range(len(dialog)):
                    # Tokenizar
                    indices = self.tokenize(dialog[j])
                    
                    # Truncar si es necesario
                    if len(indices) > self.max_seq_length:
                        indices = indices[:self.max_seq_length]
                    
                    # Padding
                    padded = indices + [self.word2idx['<PAD>']] * (self.max_seq_length - len(indices))
                    
                    features.append(padded)
                    act_labels.append(acts[j])
                    emotion_labels.append(emotions[j])
            
            return (
                torch.LongTensor(features), 
                torch.LongTensor(act_labels), 
                torch.LongTensor(emotion_labels)
            )
        
        train_features, train_acts, train_emotions = extract_features_and_labels(self.train_df)
        val_features, val_acts, val_emotions = extract_features_and_labels(self.val_df)
        test_features, test_acts, test_emotions = extract_features_and_labels(self.test_df)
        
        print(f"{Fore.GREEN}Datos preparados para clasificación.{Style.RESET_ALL}")
        print(f"  - Ejemplos de entrenamiento: {len(train_features)}")
        print(f"  - Ejemplos de validación: {len(val_features)}")
        print(f"  - Ejemplos de prueba: {len(test_features)}")
        
        return {
            'train': (train_features, train_acts, train_emotions),
            'val': (val_features, val_acts, val_emotions),
            'test': (test_features, test_acts, test_emotions)
        }
    
    def decode_sequence(self, sequence):
        """
        Convierte una secuencia de índices a texto
        Args:
            sequence: Secuencia de índices
        Returns:
            Texto decodificado
        """
        words = []
        for idx in sequence:
            if idx == self.word2idx['<PAD>'] or idx == self.word2idx['<EOS>']:
                break
            if idx == self.word2idx['<SOS>']:
                continue
            words.append(self.idx2word.get(idx.item(), '<UNK>'))
        
        return ' '.join(words)
    
    def get_stats(self):
        """
        Obtiene estadísticas del dataset
        Returns:
            Diccionario con estadísticas
        """
        print(f"{Fore.BLUE}Calculando estadísticas del dataset...{Style.RESET_ALL}")
        
        stats = {}
        
        # Contar número total de diálogos
        stats['total_dialogs'] = len(self.train_df) + len(self.test_df) + len(self.val_df)
        
        # Contar número total de turnos
        total_turns = 0
        for df in [self.train_df, self.test_df, self.val_df]:
            for dialog in df['dialog']:
                total_turns += len(dialog)
        stats['total_turns'] = total_turns
        
        # Calcular longitud promedio de turnos
        all_lengths = []
        for df in [self.train_df, self.test_df, self.val_df]:
            for dialog in df['dialog']:
                for turn in dialog:
                    all_lengths.append(len(turn.split()))
        stats['avg_turn_length'] = sum(all_lengths) / len(all_lengths)
        stats['turn_lengths'] = all_lengths
        
        # Contar distribución de actos
        act_counts = [0] * self.act_classes
        for df in [self.train_df, self.test_df, self.val_df]:
            for acts in df['act']:
                for act in acts:
                    if 0 <= act < self.act_classes:
                        act_counts[act] += 1
        stats['act_counts'] = act_counts
        
        # Contar distribución de emociones
        emotion_counts = [0] * self.emotion_classes
        for df in [self.train_df, self.test_df, self.val_df]:
            for emotions in df['emotion']:
                for emotion in emotions:
                    if 0 <= emotion < self.emotion_classes:
                        emotion_counts[emotion] += 1
        stats['emotion_counts'] = emotion_counts
        
        # Palabras más frecuentes
        word_counts = {}
        for word, idx in self.word2idx.items():
            if word not in ['<PAD>', '<UNK>', '<SOS>', '<EOS>']:
                word_counts[word] = 0
        
        for df in [self.train_df, self.test_df, self.val_df]:
            for dialog in df['dialog']:
                for turn in dialog:
                    turn = turn.strip().lower()
                    turn = re.sub(r'[^\w\s\']', ' ', turn)
                    words = turn.split()
                    for word in words:
                        if word in word_counts:
                            word_counts[word] += 1
        
        stats['most_common_words'] = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:50]
        
        print(f"{Fore.GREEN}Estadísticas calculadas correctamente.{Style.RESET_ALL}")
        return stats
    
    def visualize_stats(self, stats):
        """
        Visualiza estadísticas del dataset
        Args:
            stats: Diccionario con estadísticas
        """
        print(f"{Fore.BLUE}Generando visualizaciones de estadísticas...{Style.RESET_ALL}")
        
        # Configurar estilo de visualización
        plt.style.use('ggplot')
        sns.set(style="whitegrid")
        
        # 1. Distribución de longitud de turnos
        plt.figure(figsize=(10, 6))
        sns.histplot(stats['turn_lengths'], bins=30, kde=True)
        plt.title('Distribución de longitud de turnos de diálogo')
        plt.xlabel('Número de palabras')
        plt.ylabel('Frecuencia')
        plt.axvline(x=stats['avg_turn_length'], color='r', linestyle='--', 
                   label=f'Promedio: {stats["avg_turn_length"]:.2f} palabras')
        plt.legend()
        plt.tight_layout()
        plt.savefig('turn_length_distribution.png')
        plt.close()
        
        # 2. Distribución de actos de diálogo
        plt.figure(figsize=(10, 6))
        act_labels = ['Acto 0', 'Acto 1', 'Acto 2', 'Acto 3', 'Acto 4']
        sns.barplot(x=act_labels[:len(stats['act_counts'])], y=stats['act_counts'])
        plt.title('Distribución de actos de diálogo')
        plt.xlabel('Tipo de acto')
        plt.ylabel('Frecuencia')
        # Añadir etiquetas con valores
        for i, count in enumerate(stats['act_counts']):
            plt.text(i, count + 100, f'{count}', ha='center')
        plt.tight_layout()
        plt.savefig('act_distribution.png')
        plt.close()
        
        # 3. Distribución de emociones
        plt.figure(figsize=(12, 6))
        emotion_labels = ['Neutral', 'Alegría', 'Sorpresa', 'Tristeza', 'Enojo', 'Disgusto', 'Miedo']
        sns.barplot(x=emotion_labels[:len(stats['emotion_counts'])], y=stats['emotion_counts'])
        plt.title('Distribución de emociones en diálogos')
        plt.xlabel('Emoción')
        plt.ylabel('Frecuencia')
        # Añadir etiquetas con valores
        for i, count in enumerate(stats['emotion_counts']):
            plt.text(i, count + 100, f'{count}', ha='center')
        plt.tight_layout()
        plt.savefig('emotion_distribution.png')
        plt.close()
        
        # 4. Palabras más comunes
        plt.figure(figsize=(14, 8))
        words, counts = zip(*stats['most_common_words'][:20])  # Top 20
        sns.barplot(x=list(counts), y=list(words))
        plt.title('20 palabras más comunes en el dataset')
        plt.xlabel('Frecuencia')
        plt.ylabel('Palabra')
        # Añadir etiquetas con valores
        for i, count in enumerate(counts):
            plt.text(count + 10, i, f'{count}', va='center')
        plt.tight_layout()
        plt.savefig('common_words.png')
        plt.close()
        
        print(f"{Fore.GREEN}Visualizaciones generadas y guardadas correctamente.{Style.RESET_ALL}")

# Conjuntos de datos personalizados para PyTorch
class DialogGenerationDataset(Dataset):
    """Dataset para la tarea de generación de respuestas"""
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

class DialogClassificationDataset(Dataset):
    """Dataset para las tareas de clasificación"""
    def __init__(self, features, act_labels, emotion_labels):
        self.features = features
        self.act_labels = act_labels
        self.emotion_labels = emotion_labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.act_labels[idx], self.emotion_labels[idx]

# Implementación de la capa de codificación posicional
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100, dropout=0.1):
        """
        Inicializa la capa de codificación posicional
        Args:
            d_model: Dimensión del modelo
            max_seq_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Crear matriz de codificación posicional
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Registrar buffer (no es un parámetro pero es parte del módulo)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor de entrada [batch_size, seq_length, d_model]
        Returns:
            Tensor con codificación posicional añadida
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# Implementación de la capa de atención multi-cabezal personalizada
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Inicializa la capa de atención multi-cabezal
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            dropout: Tasa de dropout
        """
        super(MultiHeadAttentionLayer, self).__init__()
        
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Capas lineales para Q, K, V
        self.fc_query = nn.Linear(d_model, d_model)
        self.fc_key = nn.Linear(d_model, d_model)
        self.fc_value = nn.Linear(d_model, d_model)
        
        # Capa lineal final
        self.fc_out = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Factor de escala
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Tensor de consulta [batch_size, query_len, d_model]
            key: Tensor de clave [batch_size, key_len, d_model]
            value: Tensor de valor [batch_size, value_len, d_model]
            mask: Máscara opcional [batch_size, 1, 1, key_len]
        Returns:
            Tensor de salida [batch_size, query_len, d_model]
            Pesos de atención [batch_size, num_heads, query_len, key_len]
        """
        batch_size = query.shape[0]
        
        # Transformar con capas lineales
        Q = self.fc_query(query)  # [batch_size, query_len, d_model]
        K = self.fc_key(key)      # [batch_size, key_len, d_model]
        V = self.fc_value(value)  # [batch_size, value_len, d_model]
        
        # Reshape para multi-cabezal
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Calcular puntajes de atención
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # Aplicar máscara si se proporciona
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Aplicar softmax
        attention = torch.softmax(energy, dim=-1)
        
        # Aplicar dropout
        attention = self.dropout(attention)
        
        # Multiplicar por valores
        x = torch.matmul(attention, V)
        
        # Reshape de vuelta
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        
        # Capa lineal final
        x = self.fc_out(x)
        
        return x, attention

# Implementación de la capa de feed-forward
class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Inicializa la capa feed-forward
        Args:
            d_model: Dimensión del modelo
            d_ff: Dimensión interna de la capa feed-forward
            dropout: Tasa de dropout
        """
        super(PositionwiseFeedforwardLayer, self).__init__()
        
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Args:
            x: Tensor de entrada [batch_size, seq_len, d_model]
        Returns:
            Tensor de salida [batch_size, seq_len, d_model]
        """
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

# Implementación de la capa de encoder
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Inicializa una capa de encoder
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed-forward
            dropout: Tasa de dropout
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor de entrada [batch_size, src_len, d_model]
            src_mask: Máscara opcional [batch_size, 1, 1, src_len]
        Returns:
            Tensor de salida [batch_size, src_len, d_model]
            Pesos de atención [batch_size, num_heads, src_len, src_len]
        """
        # Self-attention
        _src, attention = self.self_attn(src, src, src, src_mask)
        
        # Residual connection y normalización
        src = self.norm1(src + self.dropout(_src))
        
        # Feed-forward
        _src = self.feed_forward(src)
        
        # Residual connection y normalización
        src = self.norm2(src + self.dropout(_src))
        
        return src, attention

# Implementación de la capa de decoder
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Inicializa una capa de decoder
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed-forward
            dropout: Tasa de dropout
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, enc_src, trg_mask=None, src_mask=None):
        """
        Args:
            trg: Tensor de entrada del decoder [batch_size, trg_len, d_model]
            enc_src: Tensor de salida del encoder [batch_size, src_len, d_model]
            trg_mask: Máscara para el target [batch_size, 1, trg_len, trg_len]
            src_mask: Máscara para la fuente [batch_size, 1, 1, src_len]
        Returns:
            Tensor de salida [batch_size, trg_len, d_model]
            Pesos de atención self [batch_size, num_heads, trg_len, trg_len]
            Pesos de atención encoder [batch_size, num_heads, trg_len, src_len]
        """
        # Self-attention
        _trg, self_attention = self.self_attn(trg, trg, trg, trg_mask)
        
        # Residual connection y normalización
        trg = self.norm1(trg + self.dropout(_trg))
        
        # Encoder-decoder attention
        _trg, encoder_attention = self.enc_attn(trg, enc_src, enc_src, src_mask)
        
        # Residual connection y normalización
        trg = self.norm2(trg + self.dropout(_trg))
        
        # Feed-forward
        _trg = self.feed_forward(trg)
        
        # Residual connection y normalización
        trg = self.norm3(trg + self.dropout(_trg))
        
        return trg, self_attention, encoder_attention

# Implementación del encoder completo
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.1):
        """
        Inicializa el encoder
        Args:
            input_dim: Tamaño del vocabulario de entrada
            d_model: Dimensión del modelo
            num_layers: Número de capas de encoder
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed-forward
            max_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(Encoder, self).__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_length, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, src, src_mask=None):
        """
        Args:
            src: Tensor de entrada [batch_size, src_len]
            src_mask: Máscara opcional [batch_size, 1, 1, src_len]
        Returns:
            Tensor de salida [batch_size, src_len, d_model]
            Lista de pesos de atención
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Embedding de tokens y posicional
        tok_emb = self.tok_embedding(src) * self.scale
        pos_emb = self.pos_embedding(tok_emb)
        
        # Aplicar capas de encoder
        src = pos_emb
        attentions = []
        
        for layer in self.layers:
            src, attention = layer(src, src_mask)
            attentions.append(attention)
        
        return src, attentions

# Implementación del decoder completo
class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.1):
        """
        Inicializa el decoder
        Args:
            output_dim: Tamaño del vocabulario de salida
            d_model: Dimensión del modelo
            num_layers: Número de capas de decoder
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed-forward
            max_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(Decoder, self).__init__()
        
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = PositionalEncoding(d_model, max_length, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, trg, enc_src, trg_mask=None, src_mask=None):
        """
        Args:
            trg: Tensor de entrada del decoder [batch_size, trg_len]
            enc_src: Tensor de salida del encoder [batch_size, src_len, d_model]
            trg_mask: Máscara para el target [batch_size, 1, trg_len, trg_len]
            src_mask: Máscara para la fuente [batch_size, 1, 1, src_len]
        Returns:
            Tensor de salida [batch_size, trg_len, output_dim]
            Lista de pesos de atención self
            Lista de pesos de atención encoder
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        # Embedding de tokens y posicional
        tok_emb = self.tok_embedding(trg) * self.scale
        pos_emb = self.pos_embedding(tok_emb)
        
        # Aplicar capas de decoder
        trg = pos_emb
        self_attentions = []
        encoder_attentions = []
        
        for layer in self.layers:
            trg, self_attn, enc_attn = layer(trg, enc_src, trg_mask, src_mask)
            self_attentions.append(self_attn)
            encoder_attentions.append(enc_attn)
        
        # Capa de salida
        output = self.fc_out(trg)
        
        return output, self_attentions, encoder_attentions

# Implementación del modelo Transformer completo
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        """
        Inicializa el modelo Transformer
        Args:
            encoder: Módulo encoder
            decoder: Módulo decoder
            src_pad_idx: Índice de padding para la fuente
            trg_pad_idx: Índice de padding para el target
            device: Dispositivo (CPU/GPU)
        """
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        """
        Crea una máscara para la secuencia de entrada
        Args:
            src: Tensor de entrada [batch_size, src_len]
        Returns:
            Máscara [batch_size, 1, 1, src_len]
        """
        # src_mask: [batch_size, 1, 1, src_len]
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        """
        Crea una máscara para la secuencia de salida
        Args:
            trg: Tensor de salida [batch_size, trg_len]
        Returns:
            Máscara [batch_size, 1, trg_len, trg_len]
        """
        # trg_pad_mask: [batch_size, 1, 1, trg_len]
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        # trg_len
        trg_len = trg.shape[1]
        
        # trg_sub_mask: [trg_len, trg_len]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        
        # trg_mask: [batch_size, 1, trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask
        
        return trg_mask
    
    def forward(self, src, trg):
        """
        Args:
            src: Tensor de entrada [batch_size, src_len]
            trg: Tensor de salida [batch_size, trg_len]
        Returns:
            Tensor de predicción [batch_size, trg_len, output_dim]
            Atenciones del encoder
            Atenciones self del decoder
            Atenciones encoder-decoder
        """
        # Crear máscaras
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # Pasar por el encoder
        enc_src, enc_attention = self.encoder(src, src_mask)
        
        # Pasar por el decoder
        output, self_attention, encoder_attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output, enc_attention, self_attention, encoder_attention

# Modelo de clasificación basado en Transformer
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, max_length, 
                 num_classes_act, num_classes_emotion, dropout=0.1):
        """
        Inicializa el clasificador basado en Transformer
        Args:
            input_dim: Tamaño del vocabulario de entrada
            d_model: Dimensión del modelo
            num_layers: Número de capas de encoder
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed-forward
            max_length: Longitud máxima de secuencia
            num_classes_act: Número de clases para actos
            num_classes_emotion: Número de clases para emociones
            dropout: Tasa de dropout
        """
        super(TransformerClassifier, self).__init__()
        
        # Encoder
        self.encoder = Encoder(input_dim, d_model, num_layers, num_heads, d_ff, max_length, dropout)
        
        # Capas de clasificación
        self.act_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes_act)
        )
        
        self.emotion_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes_emotion)
        )
        
        self.src_pad_idx = 0  # <PAD>
    
    def make_src_mask(self, src):
        """
        Crea una máscara para la secuencia de entrada
        Args:
            src: Tensor de entrada [batch_size, src_len]
        Returns:
            Máscara [batch_size, 1, 1, src_len]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def forward(self, src):
        """
        Args:
            src: Tensor de entrada [batch_size, src_len]
        Returns:
            Predicciones de actos [batch_size, num_classes_act]
            Predicciones de emociones [batch_size, num_classes_emotion]
        """
        # Crear máscara
        src_mask = self.make_src_mask(src)
        
        # Pasar por el encoder
        enc_src, _ = self.encoder(src, src_mask)
        
        # Obtener representación global (promedio de tokens)
        # Alternativa: usar [CLS] token o atención global
        mask = (src != self.src_pad_idx).unsqueeze(-1).float()
        enc_src = (enc_src * mask).sum(dim=1) / mask.sum(dim=1)
        
        # Clasificación
        act_preds = self.act_classifier(enc_src)
        emotion_preds = self.emotion_classifier(enc_src)
        
        return act_preds, emotion_preds

# Función para inicializar pesos
def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# Función para contar parámetros
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Función para entrenar el modelo de generación
def train_generation_model(model, dataloader, optimizer, criterion, clip, device, scheduler=None):
    """
    Entrena el modelo de generación por una época
    Args:
        model: Modelo Transformer
        dataloader: DataLoader con datos de entrenamiento
        optimizer: Optimizador
        criterion: Función de pérdida
        clip: Valor para gradient clipping
        device: Dispositivo (CPU/GPU)
        scheduler: Scheduler de tasa de aprendizaje (opcional)
    Returns:
        Pérdida promedio de la época
    """
    model.train()
    epoch_loss = 0
    
    # Barra de progreso
    progress_bar = tqdm(dataloader, desc="Entrenando", leave=False)
    
    for i, (src, trg) in enumerate(progress_bar):
        src = src.to(device)
        trg = trg.to(device)
        
        # Eliminar token EOS para la entrada al decoder
        trg_input = trg[:, :-1]
        
        # Eliminar token SOS para el objetivo
        trg_output = trg[:, 1:]
        
        # Forward pass
        output, _, _, _ = model(src, trg_input)
        
        # Reshape para calcular pérdida
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg_output = trg_output.contiguous().view(-1)
        
        # Calcular pérdida
        loss = criterion(output, trg_output)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Actualizar parámetros
        optimizer.step()
        
        # Actualizar scheduler si existe
        if scheduler is not None:
            scheduler.step()
        
        # Actualizar pérdida acumulada
        epoch_loss += loss.item()
        
        # Actualizar barra de progreso
        progress_bar.set_postfix(loss=loss.item())
    
    return epoch_loss / len(dataloader)

# Función para evaluar el modelo de generación
def evaluate_generation_model(model, dataloader, criterion, device):
    """
    Evalúa el modelo de generación
    Args:
        model: Modelo Transformer
        dataloader: DataLoader con datos de evaluación
        criterion: Función de pérdida
        device: Dispositivo (CPU/GPU)
    Returns:
        Pérdida promedio
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src = src.to(device)
            trg = trg.to(device)
            
            # Eliminar token EOS para la entrada al decoder
            trg_input = trg[:, :-1]
            
            # Eliminar token SOS para el objetivo
            trg_output = trg[:, 1:]
            
            # Forward pass
            output, _, _, _ = model(src, trg_input)
            
            # Reshape para calcular pérdida
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg_output = trg_output.contiguous().view(-1)
            
            # Calcular pérdida
            loss = criterion(output, trg_output)
            
            # Actualizar pérdida acumulada
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

# Función para generar texto con el modelo
def generate_response(model, src, processor, max_length=50, temperature=1.0):
    """
    Genera una respuesta usando el modelo
    Args:
        model: Modelo Transformer
        src: Secuencia de entrada
        processor: Procesador de diálogos
        max_length: Longitud máxima de la respuesta
        temperature: Temperatura para muestreo (1.0 = sin cambios)
    Returns:
        Texto generado
    """
    model.eval()
    
    # Preparar entrada
    src_tensor = src.unsqueeze(0).to(device)
    
    # Crear máscara
    src_mask = model.make_src_mask(src_tensor)
    
    # Codificar la fuente
    with torch.no_grad():
        enc_src, _ = model.encoder(src_tensor, src_mask)
    
    # Iniciar con token SOS
    trg_indexes = [processor.word2idx['<SOS>']]
    
    for i in range(max_length):
        # Convertir a tensor
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        
        # Crear máscara
        trg_mask = model.make_trg_mask(trg_tensor)
        
        # Decodificar
        with torch.no_grad():
            output, _, _, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        # Obtener predicción para el último token
        pred_token = output[:, -1, :]
        
        # Aplicar temperatura
        pred_token = pred_token / temperature
        
        # Convertir a probabilidades
        pred_token = F.softmax(pred_token, dim=1)
        
        # Muestrear del modelo
        pred_token_idx = torch.multinomial(pred_token, 1).item()
        
        # Añadir a la secuencia
        trg_indexes.append(pred_token_idx)
        
        # Detener si se predice EOS
        if pred_token_idx == processor.word2idx['<EOS>']:
            break
    
    # Convertir índices a texto
    trg_tokens = [processor.idx2word[i] for i in trg_indexes]
    
    # Eliminar tokens especiales
    trg_tokens = [token for token in trg_tokens if token not in ['<SOS>', '<EOS>', '<PAD>']]
    
    return ' '.join(trg_tokens)

# Función para entrenar el modelo de clasificación
def train_classification_model(model, dataloader, optimizer, criterion_act, criterion_emotion, clip, device, scheduler=None):
    """
    Entrena el modelo de clasificación por una época
    Args:
        model: Modelo TransformerClassifier
        dataloader: DataLoader con datos de entrenamiento
        optimizer: Optimizador
        criterion_act: Función de pérdida para actos
        criterion_emotion: Función de pérdida para emociones
        clip: Valor para gradient clipping
        device: Dispositivo (CPU/GPU)
        scheduler: Scheduler de tasa de aprendizaje (opcional)
    Returns:
        Pérdida promedio de la época
    """
    model.train()
    epoch_loss = 0
    
    # Barra de progreso
    progress_bar = tqdm(dataloader, desc="Entrenando", leave=False)
    
    for i, (src, act_labels, emotion_labels) in enumerate(progress_bar):
        src = src.to(device)
        act_labels = act_labels.to(device)
        emotion_labels = emotion_labels.to(device)
        
        # Forward pass
        act_preds, emotion_preds = model(src)
        
        # Calcular pérdida
        act_loss = criterion_act(act_preds, act_labels)
        emotion_loss = criterion_emotion(emotion_preds, emotion_labels)
        loss = act_loss + emotion_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Actualizar parámetros
        optimizer.step()
        
        # Actualizar scheduler si existe
        if scheduler is not None:
            scheduler.step()
        
        # Actualizar pérdida acumulada
        epoch_loss += loss.item()
        
        # Actualizar barra de progreso
        progress_bar.set_postfix(loss=loss.item())
    
    return epoch_loss / len(dataloader)

# Función para evaluar el modelo de clasificación
def evaluate_classification_model(model, dataloader, criterion_act, criterion_emotion, device):
    """
    Evalúa el modelo de clasificación
    Args:
        model: Modelo TransformerClassifier
        dataloader: DataLoader con datos de evaluación
        criterion_act: Función de pérdida para actos
        criterion_emotion: Función de pérdida para emociones
        device: Dispositivo (CPU/GPU)
    Returns:
        Pérdida promedio, métricas de actos, métricas de emociones
    """
    model.eval()
    epoch_loss = 0
    
    all_act_preds = []
    all_act_labels = []
    all_emotion_preds = []
    all_emotion_labels = []
    
    with torch.no_grad():
        for i, (src, act_labels, emotion_labels) in enumerate(dataloader):
            src = src.to(device)
            act_labels = act_labels.to(device)
            emotion_labels = emotion_labels.to(device)
            
            # Forward pass
            act_preds, emotion_preds = model(src)
            
            # Calcular pérdida
            act_loss = criterion_act(act_preds, act_labels)
            emotion_loss = criterion_emotion(emotion_preds, emotion_labels)
            loss = act_loss + emotion_loss
            
            # Actualizar pérdida acumulada
            epoch_loss += loss.item()
            
            # Guardar predicciones y etiquetas para métricas
            act_pred_classes = torch.argmax(act_preds, dim=1).cpu().numpy()
            emotion_pred_classes = torch.argmax(emotion_preds, dim=1).cpu().numpy()
            
            all_act_preds.extend(act_pred_classes)
            all_act_labels.extend(act_labels.cpu().numpy())
            all_emotion_preds.extend(emotion_pred_classes)
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
    
    # Calcular métricas
    act_accuracy = accuracy_score(all_act_labels, all_act_preds)
    act_precision, act_recall, act_f1, _ = precision_recall_fscore_support(
        all_act_labels, all_act_preds, average='weighted'
    )
    
    emotion_accuracy = accuracy_score(all_emotion_labels, all_emotion_preds)
    emotion_precision, emotion_recall, emotion_f1, _ = precision_recall_fscore_support(
        all_emotion_labels, all_emotion_preds, average='weighted'
    )
    
    act_metrics = {
        'accuracy': act_accuracy,
        'precision': act_precision,
        'recall': act_recall,
        'f1': act_f1
    }
    
    emotion_metrics = {
        'accuracy': emotion_accuracy,
        'precision': emotion_precision,
        'recall': emotion_recall,
        'f1': emotion_f1
    }
    
    return epoch_loss / len(dataloader), act_metrics, emotion_metrics

# Función para calcular métricas BLEU y ROUGE
def calculate_generation_metrics(model, dataloader, processor, device, num_samples=100):
    """
    Calcula métricas BLEU y ROUGE para el modelo de generación
    Args:
        model: Modelo Transformer
        dataloader: DataLoader con datos de evaluación
        processor: Procesador de diálogos
        device: Dispositivo (CPU/GPU)
        num_samples: Número de muestras para evaluar
    Returns:
        Diccionario con métricas
    """
    model.eval()
    bleu_scores = []
    rouge_scores = {'rouge-1': [], 'rouge-2': [], 'rouge-l': []}
    rouge = Rouge()
    
    # Función de suavizado para BLEU
    smoothie = SmoothingFunction().method1
    
    samples_processed = 0
    
    with torch.no_grad():
        for src, trg in dataloader:
            if samples_processed >= num_samples:
                break
            
            batch_size = src.size(0)
            for i in range(batch_size):
                if samples_processed >= num_samples:
                    break
                
                # Obtener entrada y referencia
                src_seq = src[i].to(device)
                trg_seq = trg[i]
                
                # Generar respuesta
                generated_text = generate_response(model, src_seq, processor)
                
                # Decodificar referencia
                reference_text = processor.decode_sequence(trg_seq)
                
                # Calcular BLEU
                reference_tokens = reference_text.split()
                generated_tokens = generated_text.split()
                
                if len(generated_tokens) > 0 and len(reference_tokens) > 0:
                    # BLEU-1, BLEU-2, BLEU-3, BLEU-4
                    bleu1 = sentence_bleu([reference_tokens], generated_tokens, 
                                         weights=(1, 0, 0, 0), smoothing_function=smoothie)
                    bleu2 = sentence_bleu([reference_tokens], generated_tokens, 
                                         weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
                    bleu3 = sentence_bleu([reference_tokens], generated_tokens, 
                                         weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
                    bleu4 = sentence_bleu([reference_tokens], generated_tokens, 
                                         weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
                    
                    bleu_scores.append({
                        'bleu-1': bleu1,
                        'bleu-2': bleu2,
                        'bleu-3': bleu3,
                        'bleu-4': bleu4
                    })
                    
                    # Calcular ROUGE
                    try:
                        rouge_score = rouge.get_scores(generated_text, reference_text)[0]
                        rouge_scores['rouge-1'].append(rouge_score['rouge-1']['f'])
                        rouge_scores['rouge-2'].append(rouge_score['rouge-2']['f'])
                        rouge_scores['rouge-l'].append(rouge_score['rouge-l']['f'])
                    except:
                        # Si hay error (por ejemplo, texto vacío), omitir esta muestra
                        pass
                
                samples_processed += 1
    
    # Calcular promedios
    avg_bleu = {
        'bleu-1': np.mean([s['bleu-1'] for s in bleu_scores]),
        'bleu-2': np.mean([s['bleu-2'] for s in bleu_scores]),
        'bleu-3': np.mean([s['bleu-3'] for s in bleu_scores]),
        'bleu-4': np.mean([s['bleu-4'] for s in bleu_scores])
    }
    
    avg_rouge = {
        'rouge-1': np.mean(rouge_scores['rouge-1']) if rouge_scores['rouge-1'] else 0,
        'rouge-2': np.mean(rouge_scores['rouge-2']) if rouge_scores['rouge-2'] else 0,
        'rouge-l': np.mean(rouge_scores['rouge-l']) if rouge_scores['rouge-l'] else 0
    }
    
    return {
        'bleu': avg_bleu,
        'rouge': avg_rouge
    }

# Función para visualizar la atención
def visualize_attention(model, src, trg, processor, layer_idx=0, head_idx=0):
    """
    Visualiza los mapas de atención del modelo
    Args:
        model: Modelo Transformer
        src: Secuencia de entrada
        trg: Secuencia de salida
        processor: Procesador de diálogos
        layer_idx: Índice de la capa a visualizar
        head_idx: Índice de la cabeza de atención a visualizar
    """
    model.eval()
    
    # Preparar entrada
    src_tensor = src.unsqueeze(0).to(device)
    trg_tensor = trg.unsqueeze(0).to(device)
    
    # Crear máscaras
    src_mask = model.make_src_mask(src_tensor)
    trg_mask = model.make_trg_mask(trg_tensor)
    
    # Forward pass
    with torch.no_grad():
        output, enc_attention, self_attention, enc_dec_attention = model(src_tensor, trg_tensor)
    
    # Obtener atención de la capa y cabeza especificada
    enc_att = enc_attention[layer_idx][0, head_idx].cpu().numpy()
    self_att = self_attention[layer_idx][0, head_idx].cpu().numpy()
    enc_dec_att = enc_dec_attention[layer_idx][0, head_idx].cpu().numpy()
    
    # Decodificar secuencias
    src_tokens = [processor.idx2word.get(idx.item(), '<UNK>') for idx in src if idx != processor.word2idx['<PAD>']]
    trg_tokens = [processor.idx2word.get(idx.item(), '<UNK>') for idx in trg if idx != processor.word2idx['<PAD>']]
    
    # Configurar visualización
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # Visualizar atención del encoder
    sns.heatmap(enc_att[:len(src_tokens), :len(src_tokens)], 
                xticklabels=src_tokens, 
                yticklabels=src_tokens, 
                cmap='viridis', 
                ax=axs[0])
    axs[0].set_title(f'Encoder Self-Attention (Layer {layer_idx}, Head {head_idx})')
    axs[0].set_xlabel('Source Tokens')
    axs[0].set_ylabel('Source Tokens')
    
    # Visualizar atención del decoder
    sns.heatmap(self_att[:len(trg_tokens), :len(trg_tokens)], 
                xticklabels=trg_tokens, 
                yticklabels=trg_tokens, 
                cmap='viridis', 
                ax=axs[1])
    axs[1].set_title(f'Decoder Self-Attention (Layer {layer_idx}, Head {head_idx})')
    axs[1].set_xlabel('Target Tokens')
    axs[1].set_ylabel('Target Tokens')
    
    # Visualizar atención encoder-decoder
    sns.heatmap(enc_dec_att[:len(trg_tokens), :len(src_tokens)], 
                xticklabels=src_tokens, 
                yticklabels=trg_tokens, 
                cmap='viridis', 
                ax=axs[2])
    axs[2].set_title(f'Encoder-Decoder Attention (Layer {layer_idx}, Head {head_idx})')
    axs[2].set_xlabel('Source Tokens')
    axs[2].set_ylabel('Target Tokens')
    
    plt.tight_layout()
    plt.savefig(f'attention_visualization_layer{layer_idx}_head{head_idx}.png')
    plt.close()

# Función para visualizar métricas de entrenamiento
def plot_training_metrics(train_losses, val_losses, train_metrics=None, val_metrics=None, metric_name=None):
    """
    Visualiza las métricas de entrenamiento
    Args:
        train_losses: Lista de pérdidas de entrenamiento
        val_losses: Lista de pérdidas de validación
        train_metrics: Lista de métricas de entrenamiento (opcional)
        val_metrics: Lista de métricas de validación (opcional)
        metric_name: Nombre de la métrica (opcional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    # Configurar visualización
    plt.figure(figsize=(12, 5))
    
    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de métrica (si se proporciona)
    if train_metrics and val_metrics and metric_name:
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_metrics, 'b-', label=f'Training {metric_name}')
        plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
        plt.title(f'Training and Validation {metric_name}')
        plt.xlabel('Epochs')
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'training_metrics_{metric_name if metric_name else "loss"}.png')
    plt.close()

# Función para visualizar matriz de confusión
def plot_confusion_matrix(y_true, y_pred, classes, title):
    """
    Visualiza la matriz de confusión
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
        classes: Lista de nombres de clases
        title: Título del gráfico
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'{title.lower().replace(" ", "_")}.png')
    plt.close()

# Función para visualizar ejemplos de generación
def visualize_generation_examples(model, test_data, processor, device, num_examples=5):
    """
    Visualiza ejemplos de generación de texto
    Args:
        model: Modelo Transformer
        test_data: Datos de prueba
        processor: Procesador de diálogos
        device: Dispositivo (CPU/GPU)
        num_examples: Número de ejemplos a visualizar
    """
    model.eval()
    
    # Seleccionar ejemplos aleatorios
    indices = np.random.choice(len(test_data), num_examples, replace=False)
    
    examples = []
    
    for idx in indices:
        src, trg = test_data[idx]
        
        # Generar respuesta
        src_tensor = src.to(device)
        generated_text = generate_response(model, src_tensor, processor)
        
        # Decodificar entrada y referencia
        input_text = processor.decode_sequence(src)
        reference_text = processor.decode_sequence(trg)
        
        examples.append({
            'input': input_text,
            'reference': reference_text,
            'generated': generated_text
        })
    
    # Crear tabla HTML para visualización
    html = "<html><head><style>"
    html += "table { border-collapse: collapse; width: 100%; }"
    html += "th, td { padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }"
    html += "th { background-color: #f2f2f2; }"
    html += "tr:hover { background-color: #f5f5f5; }"
    html += "</style></head><body>"
    html += "<h2>Ejemplos de Generación de Texto</h2>"
    html += "<table><tr><th>Entrada</th><th>Referencia</th><th>Generado</th></tr>"
    
    for ex in examples:
        html += f"<tr><td>{ex['input']}</td><td>{ex['reference']}</td><td>{ex['generated']}</td></tr>"
    
    html += "</table></body></html>"
    
    # Guardar HTML
    with open('generation_examples.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    # También imprimir ejemplos en consola
    print(f"{Fore.CYAN}Ejemplos de generación de texto:{Style.RESET_ALL}")
    for i, ex in enumerate(examples):
        print(f"\n{Fore.YELLOW}Ejemplo {i+1}:{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Entrada:{Style.RESET_ALL} {ex['input']}")
        print(f"{Fore.GREEN}Referencia:{Style.RESET_ALL} {ex['reference']}")
        print(f"{Fore.GREEN}Generado:{Style.RESET_ALL} {ex['generated']}")

# Función principal para el modelo de generación
def train_generation_transformer(processor, data, config):
    """
    Entrena y evalúa el modelo Transformer para generación
    Args:
        processor: Procesador de diálogos
        data: Datos procesados
        config: Configuración del modelo
    Returns:
        Modelo entrenado y métricas
    """
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}ENTRENAMIENTO DEL MODELO TRANSFORMER PARA GENERACIÓN DE TEXTO{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Desempaquetar datos
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    
    # Crear datasets
    train_dataset = DialogGenerationDataset(train_data[0], train_data[1])
    val_dataset = DialogGenerationDataset(val_data[0], val_data[1])
    test_dataset = DialogGenerationDataset(test_data[0], test_data[1])
    
    # Crear dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Crear modelo
    print(f"{Fore.YELLOW}Creando modelo Transformer para generación...{Style.RESET_ALL}")
    
    # Encoder
    encoder = Encoder(
        input_dim=processor.vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_length=config['max_seq_length'],
        dropout=config['dropout']
    )
    
    # Decoder
    decoder = Decoder(
        output_dim=processor.vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_length=config['max_seq_length'],
        dropout=config['dropout']
    )
    
    # Transformer completo
    model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=processor.word2idx['<PAD>'],
        trg_pad_idx=processor.word2idx['<PAD>'],
        device=device
    ).to(device)
    
    # Inicializar pesos
    model.apply(initialize_weights)
    
    # Contar parámetros
    num_params = count_parameters(model)
    print(f"{Fore.GREEN}Modelo creado con {num_params:,} parámetros entrenables.{Style.RESET_ALL}")
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Función de pérdida (ignorando padding)
    criterion = nn.CrossEntropyLoss(ignore_index=processor.word2idx['<PAD>'])
    
    # Entrenamiento
    print(f"{Fore.YELLOW}Iniciando entrenamiento...{Style.RESET_ALL}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(config['epochs']):
        print(f"\n{Fore.CYAN}Época {epoch+1}/{config['epochs']}{Style.RESET_ALL}")
        
        # Entrenar
        start_time = time.time()
        train_loss = train_generation_model(
            model, train_dataloader, optimizer, criterion, config['clip'], device
        )
        train_losses.append(train_loss)
        
        # Evaluar
        val_loss = evaluate_generation_model(model, val_dataloader, criterion, device)
        val_losses.append(val_loss)
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_generation_model.pt')
            print(f"{Fore.GREEN}Mejor modelo guardado con pérdida de validación: {best_val_loss:.4f}{Style.RESET_ALL}")
        
        # Imprimir estadísticas
        epoch_time = time.time() - start_time
        print(f"{Fore.GREEN}Época {epoch+1} completada en {epoch_time:.2f}s{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Pérdida de entrenamiento: {train_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Pérdida de validación: {val_loss:.4f}{Style.RESET_ALL}")
    
    # Visualizar métricas de entrenamiento
    plot_training_metrics(train_losses, val_losses)
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('best_generation_model.pt'))
    
    # Evaluar en conjunto de prueba
    test_loss = evaluate_generation_model(model, test_dataloader, criterion, device)
    print(f"\n{Fore.CYAN}Evaluación en conjunto de prueba:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Pérdida de prueba: {test_loss:.4f}{Style.RESET_ALL}")
    
    # Calcular métricas BLEU y ROUGE
    print(f"\n{Fore.CYAN}Calculando métricas BLEU y ROUGE...{Style.RESET_ALL}")
    generation_metrics = calculate_generation_metrics(model, test_dataloader, processor, device)
    
    print(f"\n{Fore.CYAN}Métricas de generación:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}BLEU-1: {generation_metrics['bleu']['bleu-1']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}BLEU-2: {generation_metrics['bleu']['bleu-2']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}BLEU-3: {generation_metrics['bleu']['bleu-3']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}BLEU-4: {generation_metrics['bleu']['bleu-4']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}ROUGE-1: {generation_metrics['rouge']['rouge-1']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}ROUGE-2: {generation_metrics['rouge']['rouge-2']:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}ROUGE-L: {generation_metrics['rouge']['rouge-l']:.4f}{Style.RESET_ALL}")
    
    # Visualizar ejemplos de generación
    print(f"\n{Fore.CYAN}Visualizando ejemplos de generación...{Style.RESET_ALL}")
    visualize_generation_examples(model, test_dataset, processor, device)
    
    # Visualizar atención (para un ejemplo)
    print(f"\n{Fore.CYAN}Visualizando mapas de atención...{Style.RESET_ALL}")
    src, trg = test_dataset[0]
    visualize_attention(model, src, trg, processor)
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'generation_metrics': generation_metrics
    }

# Función principal para el modelo de clasificación
def train_classification_transformer(processor, data, config):
    """
    Entrena y evalúa el modelo Transformer para clasificación
    Args:
        processor: Procesador de diálogos
        data: Datos procesados
        config: Configuración del modelo
    Returns:
        Modelo entrenado y métricas
    """
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}ENTRENAMIENTO DEL MODELO TRANSFORMER PARA CLASIFICACIÓN{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Desempaquetar datos
    train_data = data['train_classification']
    val_data = data['val_classification']
    test_data = data['test_classification']
    
    # Crear datasets
    train_dataset = DialogClassificationDataset(train_data[0], train_data[1], train_data[2])
    val_dataset = DialogClassificationDataset(val_data[0], val_data[1], val_data[2])
    test_dataset = DialogClassificationDataset(test_data[0], test_data[1], test_data[2])
    
    # Crear dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'])
    
    # Crear modelo
    print(f"{Fore.YELLOW}Creando modelo Transformer para clasificación...{Style.RESET_ALL}")
    
    model = TransformerClassifier(
        input_dim=processor.vocab_size,
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        max_length=config['max_seq_length'],
        num_classes_act=processor.act_classes,
        num_classes_emotion=processor.emotion_classes,
        dropout=config['dropout']
    ).to(device)
    
    # Inicializar pesos
    model.apply(initialize_weights)
    
    # Contar parámetros
    num_params = count_parameters(model)
    print(f"{Fore.GREEN}Modelo creado con {num_params:,} parámetros entrenables.{Style.RESET_ALL}")
    
    # Optimizador
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Funciones de pérdida
    criterion_act = nn.CrossEntropyLoss()
    criterion_emotion = nn.CrossEntropyLoss()
    
    # Entrenamiento
    print(f"{Fore.YELLOW}Iniciando entrenamiento...{Style.RESET_ALL}")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_act_accuracies = []
    val_emotion_accuracies = []
    
    for epoch in range(config['epochs']):
        print(f"\n{Fore.CYAN}Época {epoch+1}/{config['epochs']}{Style.RESET_ALL}")
        
        # Entrenar
        start_time = time.time()
        train_loss = train_classification_model(
            model, train_dataloader, optimizer, criterion_act, criterion_emotion, config['clip'], device
        )
        train_losses.append(train_loss)
        
        # Evaluar
        val_loss, act_metrics, emotion_metrics = evaluate_classification_model(
            model, val_dataloader, criterion_act, criterion_emotion, device
        )
        val_losses.append(val_loss)
        val_act_accuracies.append(act_metrics['accuracy'])
        val_emotion_accuracies.append(emotion_metrics['accuracy'])
        
        # Actualizar scheduler
        scheduler.step(val_loss)
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_classification_model.pt')
            print(f"{Fore.GREEN}Mejor modelo guardado con pérdida de validación: {best_val_loss:.4f}{Style.RESET_ALL}")
        
        # Imprimir estadísticas
        epoch_time = time.time() - start_time
        print(f"{Fore.GREEN}Época {epoch+1} completada en {epoch_time:.2f}s{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Pérdida de entrenamiento: {train_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Pérdida de validación: {val_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Precisión de actos (validación): {act_metrics['accuracy']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Precisión de emociones (validación): {emotion_metrics['accuracy']:.4f}{Style.RESET_ALL}")
    
    # Visualizar métricas de entrenamiento
    plot_training_metrics(train_losses, val_losses, val_act_accuracies, val_act_accuracies, 'Accuracy')
    
    # Cargar mejor modelo
    model.load_state_dict(torch.load('best_classification_model.pt'))
    
    # Evaluar en conjunto de prueba
    test_loss, act_metrics, emotion_metrics = evaluate_classification_model(
        model, test_dataloader, criterion_act, criterion_emotion, device
    )
    
    print(f"\n{Fore.CYAN}Evaluación en conjunto de prueba:{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Pérdida de prueba: {test_loss:.4f}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Métricas de actos:{Style.RESET_ALL}")
    print(f"  - Accuracy: {act_metrics['accuracy']:.4f}")
    print(f"  - Precision: {act_metrics['precision']:.4f}")
    print(f"  - Recall: {act_metrics['recall']:.4f}")
    print(f"  - F1-score: {act_metrics['f1']:.4f}")
    
    print(f"{Fore.GREEN}Métricas de emociones:{Style.RESET_ALL}")
    print(f"  - Accuracy: {emotion_metrics['accuracy']:.4f}")
    print(f"  - Precision: {emotion_metrics['precision']:.4f}")
    print(f"  - Recall: {emotion_metrics['recall']:.4f}")
    print(f"  - F1-score: {emotion_metrics['f1']:.4f}")
    
    # Obtener predicciones para matriz de confusión
    all_act_preds = []
    all_act_labels = []
    all_emotion_preds = []
    all_emotion_labels = []
    
    model.eval()
    with torch.no_grad():
        for src, act_labels, emotion_labels in test_dataloader:
            src = src.to(device)
            act_preds, emotion_preds = model(src)
            
            act_pred_classes = torch.argmax(act_preds, dim=1).cpu().numpy()
            emotion_pred_classes = torch.argmax(emotion_preds, dim=1).cpu().numpy()
            
            all_act_preds.extend(act_pred_classes)
            all_act_labels.extend(act_labels.cpu().numpy())
            all_emotion_preds.extend(emotion_pred_classes)
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
    
    # Visualizar matrices de confusión
    act_classes = ['Acto 0', 'Acto 1', 'Acto 2', 'Acto 3', 'Acto 4']
    emotion_classes = ['Neutral', 'Alegría', 'Sorpresa', 'Tristeza', 'Enojo', 'Disgusto', 'Miedo']
    
    plot_confusion_matrix(
        all_act_labels, all_act_preds, 
        act_classes[:processor.act_classes], 
        'Matriz de Confusión - Actos de Diálogo'
    )
    
    plot_confusion_matrix(
        all_emotion_labels, all_emotion_preds, 
        emotion_classes[:processor.emotion_classes], 
        'Matriz de Confusión - Emociones'
    )
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'act_metrics': act_metrics,
        'emotion_metrics': emotion_metrics
    }

# Función principal
def main():
    # Configuración
    config = {
        # Parámetros del dataset
        'data_path': 'dailydialog.zip',
        'train_ratio': 0.8,
        'val_ratio': 0.1,
        'test_ratio': 0.1,
        'max_seq_length': 50,
        'min_word_freq': 5,
        
        # Parámetros del modelo de generación
        'generation': {
            'd_model': 256,
            'num_layers': 3,
            'num_heads': 8,
            'd_ff': 512,
            'dropout': 0.1,
            'batch_size': 64,
            'epochs': 10,
            'learning_rate': 0.0003,
            'clip': 1.0,
            'max_seq_length': 50
        },
        
        # Parámetros del modelo de clasificación
        'classification': {
            'd_model': 256,
            'num_layers': 3,
            'num_heads': 8,
            'd_ff': 512,
            'dropout': 0.1,
            'batch_size': 64,
            'epochs': 10,
            'learning_rate': 0.0003,
            'clip': 1.0,
            'max_seq_length': 50
        }
    }
    
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}IMPLEMENTACIÓN DE TRANSFORMER PARA PROCESAMIENTO DE LENGUAJE NATURAL{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    # Verificar dispositivo
    print(f"{Fore.YELLOW}Dispositivo: {device}{Style.RESET_ALL}")
    
    # Procesar datos
    print(f"\n{Fore.CYAN}PROCESAMIENTO DE DATOS{Style.RESET_ALL}")
    processor = DialogProcessor(
        data_path=config['data_path'],
        train_ratio=config['train_ratio'],
        val_ratio=config['val_ratio'],
        test_ratio=config['test_ratio'],
        max_seq_length=config['max_seq_length'],
        min_word_freq=config['min_word_freq']
    )
    
    # Preparar datos
    data = processor.prepare_data()
    
    # Calcular y visualizar estadísticas
    stats = processor.get_stats()
    processor.visualize_stats(stats)
    
    # Entrenar modelo de generación
    generation_model, generation_metrics = train_generation_transformer(
        processor, data, config['generation']
    )
    
    # Entrenar modelo de clasificación
    classification_model, classification_metrics = train_classification_transformer(
        processor, data, config['classification']
    )
    
    # Resumen final
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}RESUMEN DE RESULTADOS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}Modelo de Generación:{Style.RESET_ALL}")
    print(f"  - Pérdida final: {generation_metrics['test_loss']:.4f}")
    print(f"  - BLEU-1: {generation_metrics['generation_metrics']['bleu']['bleu-1']:.4f}")
    print(f"  - BLEU-4: {generation_metrics['generation_metrics']['bleu']['bleu-4']:.4f}")
    print(f"  - ROUGE-L: {generation_metrics['generation_metrics']['rouge']['rouge-l']:.4f}")
    
    print(f"\n{Fore.YELLOW}Modelo de Clasificación:{Style.RESET_ALL}")
    print(f"  - Pérdida final: {classification_metrics['test_loss']:.4f}")
    print(f"  - Accuracy (actos): {classification_metrics['act_metrics']['accuracy']:.4f}")
    print(f"  - F1-score (actos): {classification_metrics['act_metrics']['f1']:.4f}")
    print(f"  - Accuracy (emociones): {classification_metrics['emotion_metrics']['accuracy']:.4f}")
    print(f"  - F1-score (emociones): {classification_metrics['emotion_metrics']['f1']:.4f}")
    
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}PROCESO COMPLETADO{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

# Clase para procesar diálogos
class DialogProcessor:
    def __init__(self, data_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, 
                 max_seq_length=50, min_word_freq=5):
        """
        Inicializa el procesador de diálogos
        Args:
            data_path: Ruta al archivo de datos
            train_ratio: Proporción de datos para entrenamiento
            val_ratio: Proporción de datos para validación
            test_ratio: Proporción de datos para prueba
            max_seq_length: Longitud máxima de secuencia
            min_word_freq: Frecuencia mínima de palabras para incluir en vocabulario
        """
        print(f"{Fore.GREEN}Inicializando procesador de diálogos...{Style.RESET_ALL}")
        self.data_path = data_path
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.max_seq_length = max_seq_length
        self.min_word_freq = min_word_freq
        
        # Cargar datos
        self.dialogs, self.act_labels, self.emotion_labels = self.load_data()
        
        # Construir vocabulario
        self.word2idx, self.idx2word, self.vocab_size = self.build_vocabulary()
        
        # Obtener número de clases
        self.act_classes = len(set(sum(self.act_labels, [])))
        self.emotion_classes = len(set(sum(self.emotion_labels, [])))
        
        print(f"{Fore.GREEN}Procesador inicializado correctamente.{Style.RESET_ALL}")
        print(f"  - Tamaño del vocabulario: {self.vocab_size}")
        print(f"  - Número de diálogos: {len(self.dialogs)}")
        print(f"  - Número de clases de actos: {self.act_classes}")
        print(f"  - Número de clases de emociones: {self.emotion_classes}")
    
    def load_data(self):
        """
        Carga los datos del archivo
        Returns:
            Tupla de (diálogos, etiquetas de actos, etiquetas de emociones)
        """
        print(f"{Fore.BLUE}Cargando datos desde: {self.data_path}{Style.RESET_ALL}")
        
        # Aquí se implementaría la carga real de datos
        # Para este ejemplo, generamos datos sintéticos
        
        # Simulación de carga de datos
        num_dialogs = 1000
        max_turns = 10
        
        dialogs = []
        act_labels = []
        emotion_labels = []
        
        for i in range(num_dialogs):
            num_turns = random.randint(3, max_turns)
            
            dialog = []
            acts = []
            emotions = []
            
            for j in range(num_turns):
                # Generar una oración aleatoria
                length = random.randint(5, 15)
                utterance = ' '.join([f'word{random.randint(1, 1000)}' for _ in range(length)])
                
                dialog.append(utterance)
                acts.append(random.randint(0, 4))  # 5 clases de actos
                emotions.append(random.randint(0, 6))  # 7 clases de emociones
            
            dialogs.append(dialog)
            act_labels.append(acts)
            emotion_labels.append(emotions)
        
        print(f"{Fore.GREEN}Datos cargados correctamente.{Style.RESET_ALL}")
        return dialogs, act_labels, emotion_labels
    
    def build_vocabulary(self):
        """
        Construye el vocabulario a partir de los diálogos
        Returns:
            Tupla de (word2idx, idx2word, vocab_size)
        """
        print(f"{Fore.BLUE}Construyendo vocabulario...{Style.RESET_ALL}")
        
        # Contar frecuencia de palabras
        word_counts = Counter()
        
        for dialog in self.dialogs:
            for utterance in dialog:
                words = utterance.lower().split()
                word_counts.update(words)
        
        # Filtrar palabras poco frecuentes
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.min_word_freq]
        
        # Crear mapeos
        word2idx = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<SOS>': 2,
            '<EOS>': 3
        }
        
        for word in filtered_words:
            if word not in word2idx:
                word2idx[word] = len(word2idx)
        
        idx2word = {idx: word for word, idx in word2idx.items()}
        vocab_size = len(word2idx)
        
        print(f"{Fore.GREEN}Vocabulario construido con {vocab_size} palabras.{Style.RESET_ALL}")
        return word2idx, idx2word, vocab_size
    
    def encode_sequence(self, sequence):
        """
        Codifica una secuencia de texto en índices
        Args:
            sequence: Lista de palabras
        Returns:
            Lista de índices
        """
        return [self.word2idx.get(word.lower(), self.word2idx['<UNK>']) 
                for word in sequence.split()]
    
    def decode_sequence(self, sequence):
        """
        Decodifica una secuencia de índices en texto
        Args:
            sequence: Lista de índices
        Returns:
            Texto decodificado
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()
        
        words = []
        for idx in sequence:
            if idx == self.word2idx['<PAD>'] or idx == self.word2idx['<EOS>']:
                break
            if idx != self.word2idx['<SOS>']:
                words.append(self.idx2word.get(idx, '<UNK>'))
        
        return ' '.join(words)
    
    def prepare_data(self):
        """
        Prepara los datos para entrenamiento, validación y prueba
        Returns:
            Diccionario con datos procesados
        """
        print(f"{Fore.BLUE}Preparando datos para entrenamiento...{Style.RESET_ALL}")
        
        # Dividir datos
        num_dialogs = len(self.dialogs)
        indices = list(range(num_dialogs))
        random.shuffle(indices)
        
        train_end = int(self.train_ratio * num_dialogs)
        val_end = train_end + int(self.val_ratio * num_dialogs)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Preparar datos para generación
        train_src, train_trg = self.prepare_generation_data([self.dialogs[i] for i in train_indices])
        val_src, val_trg = self.prepare_generation_data([self.dialogs[i] for i in val_indices])
        test_src, test_trg = self.prepare_generation_data([self.dialogs[i] for i in test_indices])
        
        # Preparar datos para clasificación
        train_inputs, train_acts, train_emotions = self.prepare_classification_data(
            [self.dialogs[i] for i in train_indices],
            [self.act_labels[i] for i in train_indices],
            [self.emotion_labels[i] for i in train_indices]
        )
        
        val_inputs, val_acts, val_emotions = self.prepare_classification_data(
            [self.dialogs[i] for i in val_indices],
            [self.act_labels[i] for i in val_indices],
            [self.emotion_labels[i] for i in val_indices]
        )
        
        test_inputs, test_acts, test_emotions = self.prepare_classification_data(
            [self.dialogs[i] for i in test_indices],
            [self.act_labels[i] for i in test_indices],
            [self.emotion_labels[i] for i in test_indices]
        )
        
        print(f"{Fore.GREEN}Datos preparados correctamente.{Style.RESET_ALL}")
        print(f"  - Ejemplos de entrenamiento (generación): {len(train_src)}")
        print(f"  - Ejemplos de validación (generación): {len(val_src)}")
        print(f"  - Ejemplos de prueba (generación): {len(test_src)}")
        print(f"  - Ejemplos de entrenamiento (clasificación): {len(train_inputs)}")
        print(f"  - Ejemplos de validación (clasificación): {len(val_inputs)}")
        print(f"  - Ejemplos de prueba (clasificación): {len(test_inputs)}")
        
        return {
            'train': (train_src, train_trg),
            'val': (val_src, val_trg),
            'test': (test_src, test_trg),
            'train_classification': (train_inputs, train_acts, train_emotions),
            'val_classification': (val_inputs, val_acts, val_emotions),
            'test_classification': (test_inputs, test_acts, test_emotions)
        }
    
    def prepare_generation_data(self, dialogs):
        """
        Prepara datos para la tarea de generación
        Args:
            dialogs: Lista de diálogos
        Returns:
            Tupla de (entradas, salidas)
        """
        sources = []
        targets = []
        
        for dialog in dialogs:
            for i in range(len(dialog) - 1):
                src = dialog[i]
                trg = dialog[i + 1]
                
                # Codificar y truncar/rellenar
                src_encoded = self.encode_sequence(src)
                trg_encoded = [self.word2idx['<SOS>']] + self.encode_sequence(trg) + [self.word2idx['<EOS>']]
                
                # Truncar si es necesario
                if len(src_encoded) > self.max_seq_length:
                    src_encoded = src_encoded[:self.max_seq_length]
                
                if len(trg_encoded) > self.max_seq_length + 1:  # +1 para <EOS>
                    trg_encoded = trg_encoded[:self.max_seq_length] + [self.word2idx['<EOS>']]
                
                # Rellenar si es necesario
                src_padded = src_encoded + [self.word2idx['<PAD>']] * (self.max_seq_length - len(src_encoded))
                trg_padded = trg_encoded + [self.word2idx['<PAD>']] * (self.max_seq_length + 2 - len(trg_encoded))
                
                sources.append(torch.tensor(src_padded, dtype=torch.long))
                targets.append(torch.tensor(trg_padded, dtype=torch.long))
        
        return sources, targets
    
    def prepare_classification_data(self, dialogs, acts, emotions):
        """
        Prepara datos para la tarea de clasificación
        Args:
            dialogs: Lista de diálogos
            acts: Lista de etiquetas de actos
            emotions: Lista de etiquetas de emociones
        Returns:
            Tupla de (entradas, etiquetas de actos, etiquetas de emociones)
        """
        inputs = []
        act_labels = []
        emotion_labels = []
        
        for dialog, dialog_acts, dialog_emotions in zip(dialogs, acts, emotions):
            for i in range(len(dialog)):
                utterance = dialog[i]
                act = dialog_acts[i]
                emotion = dialog_emotions[i]
                
                # Codificar y truncar/rellenar
                encoded = self.encode_sequence(utterance)
                
                # Truncar si es necesario
                if len(encoded) > self.max_seq_length:
                    encoded = encoded[:self.max_seq_length]
                
                # Rellenar si es necesario
                padded = encoded + [self.word2idx['<PAD>']] * (self.max_seq_length - len(encoded))
                
                inputs.append(torch.tensor(padded, dtype=torch.long))
                act_labels.append(act)
                emotion_labels.append(emotion)
        
        return inputs, torch.tensor(act_labels, dtype=torch.long), torch.tensor(emotion_labels, dtype=torch.long)
    
    def get_stats(self):
        """
        Obtiene estadísticas de los datos
        Returns:
            Diccionario con estadísticas
        """
        print(f"{Fore.BLUE}Calculando estadísticas de los datos...{Style.RESET_ALL}")
        
        # Contar longitud de diálogos
        dialog_lengths = [len(dialog) for dialog in self.dialogs]
        
        # Contar longitud de enunciados
        utterance_lengths = []
        for dialog in self.dialogs:
            for utterance in dialog:
                utterance_lengths.append(len(utterance.split()))
        
        # Contar distribución de actos
        act_distribution = Counter()
        for acts in self.act_labels:
            act_distribution.update(acts)
        
        # Contar distribución de emociones
        emotion_distribution = Counter()
        for emotions in self.emotion_labels:
            emotion_distribution.update(emotions)
        
        # Calcular estadísticas
        stats = {
            'num_dialogs': len(self.dialogs),
            'vocab_size': self.vocab_size,
            'dialog_lengths': dialog_lengths,
            'utterance_lengths': utterance_lengths,
            'act_distribution': act_distribution,
            'emotion_distribution': emotion_distribution,
            'avg_dialog_length': np.mean(dialog_lengths),
            'avg_utterance_length': np.mean(utterance_lengths),
            'max_dialog_length': max(dialog_lengths),
            'max_utterance_length': max(utterance_lengths)
        }
        
        print(f"{Fore.GREEN}Estadísticas calculadas correctamente.{Style.RESET_ALL}")
        return stats
    
    def visualize_stats(self, stats):
        """
        Visualiza estadísticas de los datos
        Args:
            stats: Diccionario con estadísticas
        """
        print(f"{Fore.BLUE}Visualizando estadísticas...{Style.RESET_ALL}")
        
        # Configurar estilo
        plt.style.use('ggplot')
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(15, 12))
        
        # 1. Distribución de longitud de diálogos
        ax1 = fig.add_subplot(2, 2, 1)
        sns.histplot(stats['dialog_lengths'], bins=20, kde=True, ax=ax1)
        ax1.set_title('Distribución de longitud de diálogos')
        ax1.set_xlabel('Número de turnos')
        ax1.set_ylabel('Frecuencia')
        ax1.axvline(stats['avg_dialog_length'], color='red', linestyle='--', 
                   label=f'Media: {stats["avg_dialog_length"]:.2f}')
        ax1.legend()
        
        # 2. Distribución de longitud de enunciados
        ax2 = fig.add_subplot(2, 2, 2)
        sns.histplot(stats['utterance_lengths'], bins=20, kde=True, ax=ax2)
        ax2.set_title('Distribución de longitud de enunciados')
        ax2.set_xlabel('Número de palabras')
        ax2.set_ylabel('Frecuencia')
        ax2.axvline(stats['avg_utterance_length'], color='red', linestyle='--', 
                   label=f'Media: {stats["avg_utterance_length"]:.2f}')
        ax2.legend()
        
        # 3. Distribución de actos de diálogo
        ax3 = fig.add_subplot(2, 2, 3)
        act_labels = [f'Acto {i}' for i in range(self.act_classes)]
        act_counts = [stats['act_distribution'][i] for i in range(self.act_classes)]
        sns.barplot(x=act_labels, y=act_counts, ax=ax3)
        ax3.set_title('Distribución de actos de diálogo')
        ax3.set_xlabel('Acto')
        ax3.set_ylabel('Frecuencia')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Distribución de emociones
        ax4 = fig.add_subplot(2, 2, 4)
        emotion_labels = ['Neutral', 'Alegría', 'Sorpresa', 'Tristeza', 'Enojo', 'Disgusto', 'Miedo']
        emotion_counts = [stats['emotion_distribution'][i] for i in range(self.emotion_classes)]
        sns.barplot(x=emotion_labels[:self.emotion_classes], y=emotion_counts, ax=ax4)
        ax4.set_title('Distribución de emociones')
        ax4.set_xlabel('Emoción')
        ax4.set_ylabel('Frecuencia')
        ax4.tick_params(axis='x', rotation=45)
        
        # Ajustar layout
        plt.tight_layout()
        plt.savefig('dataset_statistics.png')
        plt.close()
        
        print(f"{Fore.GREEN}Estadísticas visualizadas y guardadas en 'dataset_statistics.png'.{Style.RESET_ALL}")
        
        # Imprimir estadísticas adicionales
        print(f"\n{Fore.CYAN}Estadísticas del dataset:{Style.RESET_ALL}")
        print(f"  - Número de diálogos: {stats['num_dialogs']}")
        print(f"  - Tamaño del vocabulario: {stats['vocab_size']}")
        print(f"  - Longitud media de diálogo: {stats['avg_dialog_length']:.2f} turnos")
        print(f"  - Longitud media de enunciado: {stats['avg_utterance_length']:.2f} palabras")
        print(f"  - Longitud máxima de diálogo: {stats['max_dialog_length']} turnos")
        print(f"  - Longitud máxima de enunciado: {stats['max_utterance_length']} palabras")

# Datasets para PyTorch
class DialogGenerationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        """
        Dataset para la tarea de generación
        Args:
            src_sequences: Secuencias de entrada
            trg_sequences: Secuencias objetivo
        """
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences
    
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, idx):
        return self.src_sequences[idx], self.trg_sequences[idx]

class DialogClassificationDataset(Dataset):
    def __init__(self, sequences, act_labels, emotion_labels):
        """
        Dataset para la tarea de clasificación
        Args:
            sequences: Secuencias de entrada
            act_labels: Etiquetas de actos
            emotion_labels: Etiquetas de emociones
        """
        self.sequences = sequences
        self.act_labels = act_labels
        self.emotion_labels = emotion_labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.act_labels[idx], self.emotion_labels[idx]

# Componentes del modelo Transformer
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Implementación de Multi-Head Attention
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            dropout: Tasa de dropout
        """
        super(MultiHeadAttention, self).__init__()
        
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Proyecciones lineales
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.d_k])).to(device)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass
        Args:
            query: Tensor de consulta [batch_size, seq_len_q, d_model]
            key: Tensor de clave [batch_size, seq_len_k, d_model]
            value: Tensor de valor [batch_size, seq_len_v, d_model]
            mask: Máscara opcional [batch_size, 1, 1, seq_len_k]
        Returns:
            Salida y pesos de atención
        """
        batch_size = query.shape[0]
        
        # Proyectar y dividir en cabezas
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).permute(0, 2, 1, 3)
        
        # Calcular scores de atención
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # Aplicar máscara si se proporciona
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Aplicar softmax
        attention = torch.softmax(energy, dim=-1)
        
        # Aplicar dropout
        attention = self.dropout(attention)
        
        # Multiplicar por valores
        x = torch.matmul(attention, V)
        
        # Concatenar cabezas
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        
        # Proyección final
        output = self.W_o(x)
        
        return output, attention

class PositionwiseFeedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Implementación de la capa Feed Forward
        Args:
            d_model: Dimensión del modelo
            d_ff: Dimensión interna de la capa feed forward
            dropout: Tasa de dropout
        """
        super(PositionwiseFeedforward, self).__init__()
        
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Tensor de entrada [batch_size, seq_len, d_model]
        Returns:
            Tensor de salida [batch_size, seq_len, d_model]
        """
        x = self.dropout(self.activation(self.fc1(x)))
        x = self.fc2(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=5000, dropout=0.1):
        """
        Implementación de Positional Encoding
        Args:
            d_model: Dimensión del modelo
            max_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(PositionalEncoding, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Crear matriz de posición
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        
        # Registrar buffer (no es un parámetro pero es parte del módulo)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Tensor de entrada [batch_size, seq_len, d_model]
        Returns:
            Tensor con encoding posicional añadido
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Implementación de una capa del Encoder
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed forward
            dropout: Tasa de dropout
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        """
        Forward pass
        Args:
            src: Tensor de entrada [batch_size, src_len, d_model]
            src_mask: Máscara para la entrada [batch_size, 1, 1, src_len]
        Returns:
            Salida y pesos de atención
        """
        # Self Attention
        _src, attention = self.self_attn(src, src, src, src_mask)
        
        # Add & Norm
        src = self.norm1(src + self.dropout(_src))
        
        # Feed Forward
        _src = self.feed_forward(src)
        
        # Add & Norm
        src = self.norm2(src + self.dropout(_src))
        
        return src, attention

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Implementación de una capa del Decoder
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed forward
            dropout: Tasa de dropout
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.enc_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = PositionwiseFeedforward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        Forward pass
        Args:
            trg: Tensor de entrada del decoder [batch_size, trg_len, d_model]
            enc_src: Salida del encoder [batch_size, src_len, d_model]
            trg_mask: Máscara para el decoder [batch_size, 1, trg_len, trg_len]
            src_mask: Máscara para el encoder [batch_size, 1, 1, src_len]
        Returns:
            Salida y pesos de atención
        """
        # Self Attention
        _trg, self_attention = self.self_attn(trg, trg, trg, trg_mask)
        
        # Add & Norm
        trg = self.norm1(trg + self.dropout(_trg))
        
        # Encoder Attention
        _trg, encoder_attention = self.enc_attn(trg, enc_src, enc_src, src_mask)
        
        # Add & Norm
        trg = self.norm2(trg + self.dropout(_trg))
        
        # Feed Forward
        _trg = self.feed_forward(trg)
        
        # Add & Norm
        trg = self.norm3(trg + self.dropout(_trg))
        
        return trg, self_attention, encoder_attention

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.1):
        """
        Implementación del Encoder completo
        Args:
            input_dim: Tamaño del vocabulario de entrada
            d_model: Dimensión del modelo
            num_layers: Número de capas del encoder
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed forward
            max_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(Encoder, self).__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, src, src_mask):
        """
        Forward pass
        Args:
            src: Tensor de entrada [batch_size, src_len]
            src_mask: Máscara para la entrada [batch_size, 1, 1, src_len]
        Returns:
            Salida del encoder y pesos de atención
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Embedding de tokens y posición
        src = self.tok_embedding(src) * self.scale
        src = self.pos_encoding(src)
        
        # Lista para almacenar atenciones
        attentions = []
        
        # Pasar por cada capa
        for layer in self.layers:
            src, attention = layer(src, src_mask)
            attentions.append(attention)
        
        return src, attentions

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.1):
        """
        Implementación del Decoder completo
        Args:
            output_dim: Tamaño del vocabulario de salida
            d_model: Dimensión del modelo
            num_layers: Número de capas del decoder
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed forward
            max_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(Decoder, self).__init__()
        
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        Forward pass
        Args:
            trg: Tensor de entrada del decoder [batch_size, trg_len]
            enc_src: Salida del encoder [batch_size, src_len, d_model]
            trg_mask: Máscara para el decoder [batch_size, 1, trg_len, trg_len]
            src_mask: Máscara para el encoder [batch_size, 1, 1, src_len]
        Returns:
            Salida del decoder y pesos de atención
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        # Embedding de tokens y posición
        trg = self.tok_embedding(trg) * self.scale
        trg = self.pos_encoding(trg)
        
        # Listas para almacenar atenciones
        self_attentions = []
        encoder_attentions = []
        
        # Pasar por cada capa
        for layer in self.layers:
            trg, self_attention, encoder_attention = layer(trg, enc_src, trg_mask, src_mask)
            self_attentions.append(self_attention)
            encoder_attentions.append(encoder_attention)
        
        # Proyección final
        output = self.fc_out(trg)
        
        return output, self_attentions, encoder_attentions

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        """
        Implementación del modelo Transformer completo
        Args:
            encoder: Módulo encoder
            decoder: Módulo decoder
            src_pad_idx: Índice de padding para la entrada
            trg_pad_idx: Índice de padding para la salida
            device: Dispositivo (CPU/GPU)
        """
        super(Transformer, self).__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
    
    def make_src_mask(self, src):
        """
        Crea máscara para la entrada
        Args:
            src: Tensor de entrada [batch_size, src_len]
        Returns:
            Máscara [batch_size, 1, 1, src_len]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_trg_mask(self, trg):
        """
        Crea máscara para la salida (combina padding y look-ahead)
        Args:
            trg: Tensor de salida [batch_size, trg_len]
        Returns:
            Máscara [batch_size, 1, trg_len, trg_len]
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask
    
    def forward(self, src, trg):
        """
        Forward pass
        Args:
            src: Tensor de entrada [batch_size, src_len]
            trg: Tensor de salida [batch_size, trg_len]
        Returns:
            Salida del modelo y pesos de atención
        """
        # Crear máscaras
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        # Pasar por encoder
        enc_src, enc_attention = self.encoder(src, src_mask)
        
        # Pasar por decoder
        output, self_attention, enc_dec_attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        return output, enc_attention, self_attention, enc_dec_attention

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, max_length, 
                 num_classes_act, num_classes_emotion, dropout=0.1):
        """
        Implementación del modelo Transformer para clasificación
        Args:
            input_dim: Tamaño del vocabulario de entrada
            d_model: Dimensión del modelo
            num_layers: Número de capas del encoder
            num_heads: Número de cabezas de atención
            d_ff: Dimensión interna de la capa feed forward
            max_length: Longitud máxima de secuencia
            num_classes_act: Número de clases de actos
            num_classes_emotion: Número de clases de emociones
            dropout: Tasa de dropout
        """
        super(TransformerClassifier, self).__init__()
        
        # Embedding y encoding posicional
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_length, dropout)
        
        # Capas del encoder
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Pooling global
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Capas de clasificación
        self.fc_act = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes_act)
        )
        
        self.fc_emotion = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes_emotion)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, src):
        """
        Forward pass
        Args:
            src: Tensor de entrada [batch_size, src_len]
        Returns:
            Logits para actos y emociones
        """
        # Crear máscara
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        
        # Embedding de tokens y posición
        src = self.tok_embedding(src) * self.scale
        src = self.pos_encoding(src)
        
        # Pasar por capas del encoder
        for layer in self.layers:
            src, _ = layer(src, src_mask)
        
        # Pooling global
        src = src.permute(0, 2, 1)  # [batch_size, d_model, src_len]
        src = self.pool(src).squeeze(2)  # [batch_size, d_model]
        
        # Clasificación
        act_logits = self.fc_act(src)
        emotion_logits = self.fc_emotion(src)
        
        return act_logits, emotion_logits

# Funciones de entrenamiento y evaluación
def train_generation_model(model, dataloader, optimizer, criterion, clip, device):
    """
    Entrena el modelo de generación por una época
    Args:
        model: Modelo Transformer
        dataloader: DataLoader con datos de entrenamiento
        optimizer: Optimizador
        criterion: Función de pérdida
        clip: Valor para gradient clipping
        device: Dispositivo (CPU/GPU)
    Returns:
        Pérdida promedio de la época
    """
    model.train()
    epoch_loss = 0
    
    for i, (src, trg) in enumerate(dataloader):
        src, trg = src.to(device), trg.to(device)
        
        optimizer.zero_grad()
        
        # Desplazar trg para teacher forcing
        output, _, _, _ = model(src, trg[:, :-1])
        
        # Reshape para calcular pérdida
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)
        
        # Calcular pérdida
        loss = criterion(output, trg)
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Actualizar parámetros
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Mostrar progreso
        if (i + 1) % 100 == 0:
            print(f"\r  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}", end="")
    
    print()  # Nueva línea después de la barra de progreso
    return epoch_loss / len(dataloader)

def evaluate_generation_model(model, dataloader, criterion, device):
    """
    Evalúa el modelo de generación
    Args:
        model: Modelo Transformer
        dataloader: DataLoader con datos de evaluación
        criterion: Función de pérdida
        device: Dispositivo (CPU/GPU)
    Returns:
        Pérdida promedio
    """
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(dataloader):
            src, trg = src.to(device), trg.to(device)
            
            # Desplazar trg para teacher forcing
            output, _, _, _ = model(src, trg[:, :-1])
            
            # Reshape para calcular pérdida
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            
            # Calcular pérdida
            loss = criterion(output, trg)
            
            epoch_loss += loss.item()
    
    return epoch_loss / len(dataloader)

def train_classification_model(model, dataloader, optimizer, criterion_act, criterion_emotion, clip, device):
    """
    Entrena el modelo de clasificación por una época
    Args:
        model: Modelo TransformerClassifier
        dataloader: DataLoader con datos de entrenamiento
        optimizer: Optimizador
        criterion_act: Función de pérdida para actos
        criterion_emotion: Función de pérdida para emociones
        clip: Valor para gradient clipping
        device: Dispositivo (CPU/GPU)
    Returns:
        Pérdida promedio de la época
    """
    model.train()
    epoch_loss = 0
    
    for i, (src, act_labels, emotion_labels) in enumerate(dataloader):
        src = src.to(device)
        act_labels = act_labels.to(device)
        emotion_labels = emotion_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        act_logits, emotion_logits = model(src)
        
        # Calcular pérdida
        act_loss = criterion_act(act_logits, act_labels)
        emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
        loss = act_loss + emotion_loss
        
        # Backpropagation
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Actualizar parámetros
        optimizer.step()
        
        epoch_loss += loss.item()
        
        # Mostrar progreso
        if (i + 1) % 100 == 0:
            print(f"\r  Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}", end="")
    
    print()  # Nueva línea después de la barra de progreso
    return epoch_loss / len(dataloader)

def evaluate_classification_model(model, dataloader, criterion_act, criterion_emotion, device):
    """
    Evalúa el modelo de clasificación
    Args:
        model: Modelo TransformerClassifier
        dataloader: DataLoader con datos de evaluación
        criterion_act: Función de pérdida para actos
        criterion_emotion: Función de pérdida para emociones
        device: Dispositivo (CPU/GPU)
    Returns:
        Tupla de (pérdida promedio, métricas de actos, métricas de emociones)
    """
    model.eval()
    epoch_loss = 0
    
    all_act_preds = []
    all_act_labels = []
    all_emotion_preds = []
    all_emotion_labels = []
    
    with torch.no_grad():
        for src, act_labels, emotion_labels in dataloader:
            src = src.to(device)
            act_labels = act_labels.to(device)
            emotion_labels = emotion_labels.to(device)
            
            # Forward pass
            act_logits, emotion_logits = model(src)
            
            # Calcular pérdida
            act_loss = criterion_act(act_logits, act_labels)
            emotion_loss = criterion_emotion(emotion_logits, emotion_labels)
            loss = act_loss + emotion_loss
            
            epoch_loss += loss.item()
            
            # Obtener predicciones
            act_preds = torch.argmax(act_logits, dim=1)
            emotion_preds = torch.argmax(emotion_logits, dim=1)
            
            # Guardar para métricas
            all_act_preds.extend(act_preds.cpu().numpy())
            all_act_labels.extend(act_labels.cpu().numpy())
            all_emotion_preds.extend(emotion_preds.cpu().numpy())
            all_emotion_labels.extend(emotion_labels.cpu().numpy())
    
    # Calcular métricas
    act_metrics = calculate_metrics(all_act_labels, all_act_preds)
    emotion_metrics = calculate_metrics(all_emotion_labels, all_emotion_preds)
    
    return epoch_loss / len(dataloader), act_metrics, emotion_metrics

def calculate_metrics(y_true, y_pred):
    """
    Calcula métricas de clasificación
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones
    Returns:
        Diccionario con métricas
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def calculate_generation_metrics(model, dataloader, processor, device):
    """
    Calcula métricas para generación de texto (BLEU, ROUGE)
    Args:
        model: Modelo Transformer
        dataloader: DataLoader con datos de evaluación
        processor: Procesador de diálogos
        device: Dispositivo (CPU/GPU)
    Returns:
        Diccionario con métricas
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge import Rouge
    
    model.eval()
    
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            
            # Generar respuesta
            generated = generate_response(model, src, processor, device)
            
            # Decodificar respuesta generada y referencia
            for i in range(len(src)):
                reference = processor.decode_sequence(trg[i][1:])  # Ignorar <SOS>
                hypothesis = generated[i]
                
                references.append(reference)
                hypotheses.append(hypothesis)
    
    # Calcular BLEU
    smoothie = SmoothingFunction().method1
    bleu_1 = 0.0
    bleu_2 = 0.0
    bleu_3 = 0.0
    bleu_4 = 0.0
    
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        
        if len(hyp_tokens) == 0:
            continue
        
        bleu_1 += sentence_bleu([ref_tokens], hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothie)
        bleu_2 += sentence_bleu([ref_tokens], hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
        bleu_3 += sentence_bleu([ref_tokens], hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothie)
        bleu_4 += sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)
    
    num_samples = len(references)
    bleu_1 /= num_samples
    bleu_2 /= num_samples
    bleu_3 /= num_samples
    bleu_4 /= num_samples
    
    # Calcular ROUGE
    rouge = Rouge()
    rouge_scores = {'rouge-1': 0.0, 'rouge-2': 0.0, 'rouge-l': 0.0}
    valid_samples = 0
    
    for ref, hyp in zip(references, hypotheses):
        if len(hyp.strip()) == 0:
            continue
        
        try:
            scores = rouge.get_scores(hyp, ref)[0]
            rouge_scores['rouge-1'] += scores['rouge-1']['f']
            rouge_scores['rouge-2'] += scores['rouge-2']['f']
            rouge_scores['rouge-l'] += scores['rouge-l']['f']
            valid_samples += 1
        except:
            continue
    
    if valid_samples > 0:
        rouge_scores['rouge-1'] /= valid_samples
        rouge_scores['rouge-2'] /= valid_samples
        rouge_scores['rouge-l'] /= valid_samples
    
    return {
        'bleu': {
            'bleu-1': bleu_1,
            'bleu-2': bleu_2,
            'bleu-3': bleu_3,
            'bleu-4': bleu_4
        },
        'rouge': rouge_scores
    }

def generate_response(model, src, processor, device, max_length=50):
    """
    Genera respuestas usando el modelo
    Args:
        model: Modelo Transformer
        src: Tensor de entrada [batch_size, src_len]
        processor: Procesador de diálogos
        device: Dispositivo (CPU/GPU)
        max_length: Longitud máxima de generación
    Returns:
        Lista de respuestas generadas
    """
    model.eval()
    batch_size = src.shape[0]
    
    # Crear máscara para src
    src_mask = model.make_src_mask(src)
    
    # Codificar src
    with torch.no_grad():
        enc_src, _ = model.encoder(src, src_mask)
    
    # Inicializar primeros tokens como <SOS>
    trg_indexes = torch.ones(batch_size, 1).long().to(device) * processor.word2idx['<SOS>']
    
    # Generar tokens uno a uno
    for i in range(max_length):
        trg_mask = model.make_trg_mask(trg_indexes)
        
        with torch.no_grad():
            output, _, _, _ = model.decoder(trg_indexes, enc_src, trg_mask, src_mask)
        
        # Obtener último token predicho
        pred_token = output[:, -1, :].argmax(dim=1).unsqueeze(1)
        
        # Añadir a secuencia
        trg_indexes = torch.cat((trg_indexes, pred_token), dim=1)
        
        # Detener si todos han generado <EOS>
        if (pred_token == processor.word2idx['<EOS>']).sum() == batch_size:
            break
    
    # Convertir a texto
    generated_responses = []
    for i in range(batch_size):
        generated_responses.append(processor.decode_sequence(trg_indexes[i]))
    
    return generated_responses

def visualize_generation_examples(model, dataset, processor, device, num_examples=5):
    """
    Visualiza ejemplos de generación
    Args:
        model: Modelo Transformer
        dataset: Dataset de evaluación
        processor: Procesador de diálogos
        device: Dispositivo (CPU/GPU)
        num_examples: Número de ejemplos a visualizar
    """
    model.eval()
    
    # Seleccionar ejemplos aleatorios
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    
    for idx in indices:
        src, trg = dataset[idx]
        src = src.unsqueeze(0).to(device)
        
        # Generar respuesta
        generated = generate_response(model, src, processor, device)[0]
        
        # Decodificar entrada y referencia
        input_text = processor.decode_sequence(src[0])
        reference = processor.decode_sequence(trg[1:])  # Ignorar <SOS>
        
        print(f"\n{Fore.CYAN}Ejemplo de generación:{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Entrada: {Fore.WHITE}{input_text}")
        print(f"{Fore.YELLOW}Referencia: {Fore.WHITE}{reference}")
        print(f"{Fore.YELLOW}Generado: {Fore.WHITE}{generated}")

def visualize_attention(model, src, trg, processor):
    """
    Visualiza mapas de atención
    Args:
        model: Modelo Transformer
        src: Tensor de entrada
        trg: Tensor objetivo
        processor: Procesador de diálogos
    """
    model.eval()
    
    # Preparar entrada
    src = src.unsqueeze(0).to(device)
    trg = trg.unsqueeze(0).to(device)
    
    # Crear máscaras
    src_mask = model.make_src_mask(src)
    trg_mask = model.make_trg_mask(trg[:, :-1])
    
    # Forward pass
    with torch.no_grad():
        output, enc_attention, dec_self_attention, dec_enc_attention = model(src, trg[:, :-1])
    
    # Obtener último mapa de atención
    enc_att = enc_attention[-1][0].cpu().numpy()
    dec_self_att = dec_self_attention[-1][0].cpu().numpy()
    dec_enc_att = dec_enc_attention[-1][0].cpu().numpy()
    
    # Decodificar tokens
    src_tokens = [processor.idx2word.get(idx.item(), '<UNK>') for idx in src[0] if idx.item() != processor.word2idx['<PAD>']]
    trg_tokens = [processor.idx2word.get(idx.item(), '<UNK>') for idx in trg[0, :-1] if idx.item() != processor.word2idx['<PAD>']]
    
    # Visualizar atención del encoder
    plt.figure(figsize=(10, 8))
    sns.heatmap(enc_att[:len(src_tokens), :len(src_tokens)], 
                xticklabels=src_tokens, 
                yticklabels=src_tokens, 
                cmap='viridis')
    plt.title('Atención del Encoder')
    plt.xlabel('Tokens de entrada')
    plt.ylabel('Tokens de entrada')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('encoder_attention.png')
    plt.close()
    
    # Visualizar auto-atención del decoder
    plt.figure(figsize=(10, 8))
    sns.heatmap(dec_self_att[:len(trg_tokens), :len(trg_tokens)], 
                xticklabels=trg_tokens, 
                yticklabels=trg_tokens, 
                cmap='viridis')
    plt.title('Auto-atención del Decoder')
    plt.xlabel('Tokens de salida')
    plt.ylabel('Tokens de salida')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('decoder_self_attention.png')
    plt.close()
    
    # Visualizar atención encoder-decoder
    plt.figure(figsize=(10, 8))
    sns.heatmap(dec_enc_att[:len(trg_tokens), :len(src_tokens)], 
                xticklabels=src_tokens, 
                yticklabels=trg_tokens, 
                cmap='viridis')
    plt.title('Atención Encoder-Decoder')
    plt.xlabel('Tokens de entrada')
    plt.ylabel('Tokens de salida')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('encoder_decoder_attention.png')
    plt.close()
    
    print(f"{Fore.GREEN}Mapas de atención guardados como imágenes.{Style.RESET_ALL}")

def visualize_training_history(history):
    """
    Visualiza el historial de entrenamiento
    Args:
        history: Diccionario con historial de entrenamiento
    """
    plt.figure(figsize=(12, 8))
    
    # Gráfico de pérdida
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Validation', marker='s')
    plt.title('Pérdida durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Gráfico de accuracy para actos
    plt.subplot(2, 2, 2)
    plt.plot(history['act_accuracy'], marker='o')
    plt.title('Accuracy de actos de diálogo')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Gráfico de accuracy para emociones
    plt.subplot(2, 2, 3)
    plt.plot(history['emotion_accuracy'], marker='o')
    plt.title('Accuracy de emociones')
    plt.xlabel('Época')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Gráfico de BLEU score
    plt.subplot(2, 2, 4)
    plt.plot(history['bleu'], marker='o')
    plt.title('BLEU-4 Score')
    plt.xlabel('Época')
    plt.ylabel('BLEU-4')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print(f"{Fore.GREEN}Historial de entrenamiento visualizado y guardado como 'training_history.png'.{Style.RESET_ALL}")

def main():
    """
    Función principal
    """
    # Configuración
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    
    # Parámetros
    data_path = 'dialogs.json'  # Ruta al archivo de datos (simulado)
    batch_size = 32
    d_model = 256
    num_heads = 8
    num_encoder_layers = 3
    num_decoder_layers = 3
    d_ff = 512
    dropout = 0.1
    max_seq_length = 50
    epochs = 10
    learning_rate = 0.0003
    clip = 1.0
    
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}PROCESAMIENTO DE DIÁLOGOS CON TRANSFORMERS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    
    print(f"\n{Fore.CYAN}CONFIGURACIÓN:{Style.RESET_ALL}")
    print(f"  - Dispositivo: {device}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Dimensión del modelo: {d_model}")
    print(f"  - Número de cabezas de atención: {num_heads}")
    print(f"  - Capas del encoder: {num_encoder_layers}")
    print(f"  - Capas del decoder: {num_decoder_layers}")
    print(f"  - Dimensión feed-forward: {d_ff}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Longitud máxima de secuencia: {max_seq_length}")
    print(f"  - Épocas: {epochs}")
    print(f"  - Tasa de aprendizaje: {learning_rate}")
    
    # Procesar datos
    print(f"\n{Fore.CYAN}PROCESAMIENTO DE DATOS{Style.RESET_ALL}")
    processor = DialogProcessor(data_path, max_seq_length=max_seq_length)
    
    # Obtener y visualizar estadísticas
    stats = processor.get_stats()
    processor.visualize_stats(stats)
    
    # Preparar datos
    data = processor.prepare_data()
    
    # Crear datasets
    train_dataset = DialogGenerationDataset(data['train'][0], data['train'][1])
    val_dataset = DialogGenerationDataset(data['val'][0], data['val'][1])
    test_dataset = DialogGenerationDataset(data['test'][0], data['test'][1])
    
    train_classification_dataset = DialogClassificationDataset(
        data['train_classification'][0], 
        data['train_classification'][1], 
        data['train_classification'][2]
    )
    val_classification_dataset = DialogClassificationDataset(
        data['val_classification'][0], 
        data['val_classification'][1], 
        data['val_classification'][2]
    )
    test_classification_dataset = DialogClassificationDataset(
        data['test_classification'][0], 
        data['test_classification'][1], 
        data['test_classification'][2]
    )
    
    # Crear dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    train_classification_dataloader = DataLoader(train_classification_dataset, batch_size=batch_size, shuffle=True)
    val_classification_dataloader = DataLoader(val_classification_dataset, batch_size=batch_size)
    test_classification_dataloader = DataLoader(test_classification_dataset, batch_size=batch_size)
    
    # Crear modelos
    print(f"\n{Fore.CYAN}CREACIÓN DE MODELOS{Style.RESET_ALL}")
    
    # Modelo de generación
    print(f"{Fore.YELLOW}Creando modelo de generación...{Style.RESET_ALL}")
    encoder = Encoder(
        input_dim=processor.vocab_size,
        d_model=d_model,
        num_layers=num_encoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_length=max_seq_length,
        dropout=dropout
    )
    
    decoder = Decoder(
        output_dim=processor.vocab_size,
        d_model=d_model,
        num_layers=num_decoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_length=max_seq_length,
        dropout=dropout
    )
    
    generation_model = Transformer(
        encoder=encoder,
        decoder=decoder,
        src_pad_idx=processor.word2idx['<PAD>'],
        trg_pad_idx=processor.word2idx['<PAD>'],
        device=device
    ).to(device)
    
    # Inicializar parámetros
    for p in generation_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Modelo de clasificación
    print(f"{Fore.YELLOW}Creando modelo de clasificación...{Style.RESET_ALL}")
    classification_model = TransformerClassifier(
        input_dim=processor.vocab_size,
        d_model=d_model,
        num_layers=num_encoder_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        max_length=max_seq_length,
        num_classes_act=processor.act_classes,
        num_classes_emotion=processor.emotion_classes,
        dropout=dropout
    ).to(device)
    
    # Inicializar parámetros
    for p in classification_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Optimizadores y funciones de pérdida
    generation_optimizer = optim.Adam(generation_model.parameters(), lr=learning_rate)
    classification_optimizer = optim.Adam(classification_model.parameters(), lr=learning_rate)
    
    generation_criterion = nn.CrossEntropyLoss(ignore_index=processor.word2idx['<PAD>'])
    classification_criterion_act = nn.CrossEntropyLoss()
    classification_criterion_emotion = nn.CrossEntropyLoss()
    
    # Mostrar resumen de los modelos
    print(f"\n{Fore.YELLOW}Resumen del modelo de generación:{Style.RESET_ALL}")
    total_params = sum(p.numel() for p in generation_model.parameters())
    trainable_params = sum(p.numel() for p in generation_model.parameters() if p.requires_grad)
    print(f"  - Parámetros totales: {total_params:,}")
    print(f"  - Parámetros entrenables: {trainable_params:,}")
    
    print(f"\n{Fore.YELLOW}Resumen del modelo de clasificación:{Style.RESET_ALL}")
    total_params = sum(p.numel() for p in classification_model.parameters())
    trainable_params = sum(p.numel() for p in classification_model.parameters() if p.requires_grad)
    print(f"  - Parámetros totales: {total_params:,}")
    print(f"  - Parámetros entrenables: {trainable_params:,}")
    
    # Entrenamiento
    print(f"\n{Fore.CYAN}ENTRENAMIENTO DE MODELOS{Style.RESET_ALL}")
    
    # Historial de entrenamiento
    history = {
        'train_loss': [],
        'val_loss': [],
        'act_accuracy': [],
        'emotion_accuracy': [],
        'bleu': []
    }
    
    # Entrenamiento del modelo de generación
    print(f"\n{Fore.YELLOW}Entrenando modelo de generación...{Style.RESET_ALL}")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{Fore.GREEN}Época {epoch+1}/{epochs}{Style.RESET_ALL}")
        
        start_time = time.time()
        
        # Entrenar
        train_loss = train_generation_model(
            generation_model, 
            train_dataloader, 
            generation_optimizer, 
            generation_criterion, 
            clip, 
            device
        )
        
        # Evaluar
        val_loss = evaluate_generation_model(
            generation_model, 
            val_dataloader, 
            generation_criterion, 
            device
        )
        
        # Calcular métricas de generación
        generation_metrics = calculate_generation_metrics(
            generation_model, 
            val_dataloader, 
            processor, 
            device
        )
        
        # Actualizar historial
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['bleu'].append(generation_metrics['bleu']['bleu-4'])
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(generation_model.state_dict(), 'best_generation_model.pt')
            print(f"{Fore.GREEN}Modelo guardado como 'best_generation_model.pt'{Style.RESET_ALL}")
        
        # Mostrar resultados
        epoch_time = time.time() - start_time
        print(f"{Fore.GREEN}Tiempo: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}BLEU-1: {generation_metrics['bleu']['bleu-1']:.4f} | BLEU-4: {generation_metrics['bleu']['bleu-4']:.4f} | ROUGE-L: {generation_metrics['rouge']['rouge-l']:.4f}{Style.RESET_ALL}")
        
        # Mostrar ejemplos
        if (epoch + 1) % 2 == 0:
            visualize_generation_examples(generation_model, val_dataset, processor, device, num_examples=2)
    
    # Entrenamiento del modelo de clasificación
    print(f"\n{Fore.YELLOW}Entrenando modelo de clasificación...{Style.RESET_ALL}")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n{Fore.GREEN}Época {epoch+1}/{epochs}{Style.RESET_ALL}")
        
        start_time = time.time()
        
        # Entrenar
        train_loss = train_classification_model(
            classification_model, 
            train_classification_dataloader, 
            classification_optimizer, 
            classification_criterion_act, 
            classification_criterion_emotion, 
            clip, 
            device
        )
        
        # Evaluar
        val_loss, act_metrics, emotion_metrics = evaluate_classification_model(
            classification_model, 
            val_classification_dataloader, 
            classification_criterion_act, 
            classification_criterion_emotion, 
            device
        )
        
        # Actualizar historial
        history['act_accuracy'].append(act_metrics['accuracy'])
        history['emotion_accuracy'].append(emotion_metrics['accuracy'])
        
        # Guardar mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(classification_model.state_dict(), 'best_classification_model.pt')
            print(f"{Fore.GREEN}Modelo guardado como 'best_classification_model.pt'{Style.RESET_ALL}")
        
        # Mostrar resultados
        epoch_time = time.time() - start_time
        print(f"{Fore.GREEN}Tiempo: {epoch_time:.2f}s | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Act Accuracy: {act_metrics['accuracy']:.4f} | Act F1: {act_metrics['f1']:.4f}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Emotion Accuracy: {emotion_metrics['accuracy']:.4f} | Emotion F1: {emotion_metrics['f1']:.4f}{Style.RESET_ALL}")
    
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math
import time
from collections import Counter
from colorama import Fore, Style, init

# Inicializar colorama para colores en la terminal
init(autoreset=True)

# Verificar si CUDA está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"{Fore.CYAN}Utilizando dispositivo: {Fore.YELLOW}{device}{Style.RESET_ALL}")

class DialogProcessor:
    def __init__(self, file_path, max_seq_length=50, min_word_freq=2):
        """
        Inicializa el procesador de diálogos
        Args:
            file_path: Ruta al archivo de diálogos
            max_seq_length: Longitud máxima de secuencia
            min_word_freq: Frecuencia mínima de palabras para incluir en vocabulario
        """
        print(f"{Fore.GREEN}Inicializando procesador de diálogos...{Style.RESET_ALL}")
        self.file_path = file_path
        self.max_seq_length = max_seq_length
        self.min_word_freq = min_word_freq
        
        # Cargar datos
        self.dialogs, self.act_labels, self.emotion_labels = self.load_data()
        
        # Construir vocabulario
        self.build_vocab()
        
        # Número de clases
        self.act_classes = max([max(acts) for acts in self.act_labels]) + 1
        self.emotion_classes = max([max(emotions) for emotions in self.emotion_labels]) + 1
        
        print(f"{Fore.GREEN}Procesador de diálogos inicializado correctamente.{Style.RESET_ALL}")
        print(f"  - Diálogos cargados: {len(self.dialogs)}")
        print(f"  - Tamaño del vocabulario: {self.vocab_size}")
        print(f"  - Clases de actos: {self.act_classes}")
        print(f"  - Clases de emociones: {self.emotion_classes}")
    
    def load_data(self):
        """
        Carga los datos de diálogos
        Returns:
            Tupla de (diálogos, etiquetas de actos, etiquetas de emociones)
        """
        try:
            print(f"{Fore.BLUE}Cargando diálogos desde: {self.file_path}{Style.RESET_ALL}")
            
            # En un caso real, cargaríamos desde un archivo
            # with open(self.file_path, 'r', encoding='utf-8') as f:
            #     data = json.load(f)
            
            # Para este ejemplo, generamos datos sintéticos
            dialogs = []
            act_labels = []
            emotion_labels = []
            
            # Vocabulario básico para generar diálogos sintéticos
            vocabulary = [
                "hola", "cómo", "estás", "bien", "gracias", "y", "tú", "qué", "tal", 
                "me", "llamo", "encantado", "conocerte", "adiós", "hasta", "luego",
                "quiero", "preguntar", "sobre", "algo", "importante", "interesante",
                "no", "sí", "quizás", "tal", "vez", "por", "supuesto", "claro",
                "feliz", "triste", "enojado", "sorprendido", "asustado", "confundido",
                "ayuda", "por", "favor", "necesito", "información", "gracias", "de", "nada"
            ]
            
            # Generar diálogos sintéticos
            for i in range(1000):  # 1000 diálogos
                dialog_length = random.randint(3, 10)  # Entre 3 y 10 turnos
                dialog = []
                acts = []
                emotions = []
                
                for j in range(dialog_length):
                    # Generar un enunciado aleatorio
                    utterance_length = random.randint(3, 15)  # Entre 3 y 15 palabras
                    utterance = [random.choice(vocabulary) for _ in range(utterance_length)]
                    dialog.append(" ".join(utterance))
                    
                    # Asignar etiqueta de acto aleatoria (0-5)
                    acts.append(random.randint(0, 5))
                    
                    # Asignar etiqueta de emoción aleatoria (0-6)
                    emotions.append(random.randint(0, 6))
                
                dialogs.append(dialog)
                act_labels.append(acts)
                emotion_labels.append(emotions)
            
            print(f"{Fore.GREEN}Datos cargados correctamente. {len(dialogs)} diálogos generados.{Style.RESET_ALL}")
            return dialogs, act_labels, emotion_labels
            
        except Exception as e:
            print(f"{Fore.RED}Error al cargar los datos: {e}{Style.RESET_ALL}")
            # Devolver datos vacíos en caso de error
            return [], [], []
    
    def build_vocab(self):
        """
        Construye el vocabulario a partir de los diálogos
        """
        print(f"{Fore.BLUE}Construyendo vocabulario...{Style.RESET_ALL}")
        
        # Contar frecuencia de palabras
        word_counts = Counter()
        for dialog in self.dialogs:
            for utterance in dialog:
                word_counts.update(utterance.split())
        
        # Filtrar palabras poco frecuentes
        filtered_words = [word for word, count in word_counts.items() 
                         if count >= self.min_word_freq]
        
        # Crear vocabulario con tokens especiales
        self.word2idx = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3
        }
        
        # Añadir palabras filtradas
        for word in filtered_words:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        
        # Crear mapeo inverso
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        
        # Tamaño del vocabulario
        self.vocab_size = len(self.word2idx)
        
        print(f"{Fore.GREEN}Vocabulario construido. Tamaño: {self.vocab_size} palabras{Style.RESET_ALL}")
    
    def encode_sequence(self, sequence):
        """
        Codifica una secuencia de texto en índices
        Args:
            sequence: Secuencia de texto
        Returns:
            Lista de índices
        """
        words = sequence.split()
        # Truncar si es necesario
        if len(words) > self.max_seq_length - 2:  # -2 para <SOS> y <EOS>
            words = words[:self.max_seq_length - 2]
        
        # Codificar
        encoded = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        
        # Añadir tokens especiales
        encoded = [self.word2idx['<SOS>']] + encoded + [self.word2idx['<EOS>']]
        
        return encoded
    
    def decode_sequence(self, sequence):
        """
        Decodifica una secuencia de índices en texto
        Args:
            sequence: Secuencia de índices (tensor o lista)
        Returns:
            Texto decodificado
        """
        if isinstance(sequence, torch.Tensor):
            sequence = sequence.cpu().numpy()
        
        # Filtrar tokens especiales
        words = []
        for idx in sequence:
            if idx == self.word2idx['<PAD>'] or idx == self.word2idx['<SOS>']:
                continue
            if idx == self.word2idx['<EOS>']:
                break
            words.append(self.idx2word.get(idx, '<UNK>'))
        
        return ' '.join(words)
    
    def get_stats(self):
        """
        Obtiene estadísticas de los diálogos
        Returns:
            Diccionario con estadísticas
        """
        print(f"{Fore.BLUE}Calculando estadísticas de los diálogos...{Style.RESET_ALL}")
        
        # Longitud de diálogos
        dialog_lengths = [len(dialog) for dialog in self.dialogs]
        
        # Longitud de enunciados
        utterance_lengths = []
        for dialog in self.dialogs:
            for utterance in dialog:
                utterance_lengths.append(len(utterance.split()))
        
        # Distribución de actos de diálogo
        act_distribution = Counter()
        for acts in self.act_labels:
            act_distribution.update(acts)
        
        # Distribución de emociones
        emotion_distribution = Counter()
        for emotions in self.emotion_labels:
            emotion_distribution.update(emotions)
        
        # Palabras más comunes
        word_counts = Counter()
        for dialog in self.dialogs:
            for utterance in dialog:
                word_counts.update(utterance.split())
        
        most_common_words = word_counts.most_common(20)
        
        return {
            'dialog_lengths': dialog_lengths,
            'utterance_lengths': utterance_lengths,
            'act_distribution': act_distribution,
            'emotion_distribution': emotion_distribution,
            'most_common_words': most_common_words,
            'total_dialogs': len(self.dialogs),
            'total_utterances': sum(dialog_lengths),
            'avg_dialog_length': sum(dialog_lengths) / len(self.dialogs) if self.dialogs else 0,
            'avg_utterance_length': sum(utterance_lengths) / len(utterance_lengths) if utterance_lengths else 0
        }
    
    def visualize_stats(self, stats):
        """
        Visualiza estadísticas de los diálogos
        Args:
            stats: Diccionario con estadísticas
        """
        print(f"\n{Fore.CYAN}--- ESTADÍSTICAS DE LOS DIÁLOGOS ---{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}Total de diálogos: {Fore.WHITE}{stats['total_dialogs']}")
        print(f"{Fore.YELLOW}Total de enunciados: {Fore.WHITE}{stats['total_utterances']}")
        print(f"{Fore.YELLOW}Longitud media de diálogo: {Fore.WHITE}{stats['avg_dialog_length']:.2f} enunciados")
        print(f"{Fore.YELLOW}Longitud media de enunciado: {Fore.WHITE}{stats['avg_utterance_length']:.2f} palabras")
        
        # Visualizar distribución de longitud de diálogos
        plt.figure(figsize=(12, 6))
        sns.histplot(stats['dialog_lengths'], bins=20, kde=True)
        plt.title('Distribución de longitud de diálogos')
        plt.xlabel('Número de enunciados por diálogo')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig('dialog_lengths.png')
        plt.close()
        
        # Visualizar distribución de longitud de enunciados
        plt.figure(figsize=(12, 6))
        sns.histplot(stats['utterance_lengths'], bins=30, kde=True)
        plt.title('Distribución de longitud de enunciados')
        plt.xlabel('Número de palabras por enunciado')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.savefig('utterance_lengths.png')
        plt.close()
        
        # Visualizar distribución de actos de diálogo
        plt.figure(figsize=(12, 6))
        acts, counts = zip(*sorted(stats['act_distribution'].items()))
        plt.bar(acts, counts, color='skyblue')
        plt.title('Distribución de actos de diálogo')
        plt.xlabel('Tipo de acto')
        plt.ylabel('Frecuencia')
        plt.xticks(acts)
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig('act_distribution.png')
        plt.close()
        
        # Visualizar distribución de emociones
        plt.figure(figsize=(12, 6))
        emotions, counts = zip(*sorted(stats['emotion_distribution'].items()))
        plt.bar(emotions, counts, color='salmon')
        plt.title('Distribución de emociones')
        plt.xlabel('Tipo de emoción')
        plt.ylabel('Frecuencia')
        plt.xticks(emotions)
        plt.grid(True, axis='y', alpha=0.3)
        plt.savefig('emotion_distribution.png')
        plt.close()
        
        # Visualizar palabras más comunes
        plt.figure(figsize=(14, 8))
        words, counts = zip(*stats['most_common_words'])
        plt.barh(range(len(words)), counts, color='lightgreen')
        plt.yticks(range(len(words)), words)
        plt.title('Palabras más comunes')
        plt.xlabel('Frecuencia')
        plt.ylabel('Palabra')
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig('common_words.png')
        plt.close()
        
        print(f"{Fore.GREEN}Visualizaciones guardadas como imágenes.{Style.RESET_ALL}")
    
    def prepare_data(self):
        """
        Prepara los datos para entrenamiento, validación y prueba
        Returns:
            Diccionario con datos procesados
        """
        print(f"{Fore.BLUE}Preparando datos para entrenamiento...{Style.RESET_ALL}")
        
        # Dividir en conjuntos de entrenamiento, validación y prueba
        num_dialogs = len(self.dialogs)
        indices = list(range(num_dialogs))
        random.shuffle(indices)
        
        train_size = int(0.7 * num_dialogs)
        val_size = int(0.15 * num_dialogs)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        # Preparar datos para generación
        train_src, train_trg = self.prepare_generation_data([self.dialogs[i] for i in train_indices])
        val_src, val_trg = self.prepare_generation_data([self.dialogs[i] for i in val_indices])
        test_src, test_trg = self.prepare_generation_data([self.dialogs[i] for i in test_indices])
        
        # Preparar datos para clasificación
        train_class_src, train_acts, train_emotions = self.prepare_classification_data(
            [self.dialogs[i] for i in train_indices],
            [self.act_labels[i] for i in train_indices],
            [self.emotion_labels[i] for i in train_indices]
        )
        
        val_class_src, val_acts, val_emotions = self.prepare_classification_data(
            [self.dialogs[i] for i in val_indices],
            [self.act_labels[i] for i in val_indices],
            [self.emotion_labels[i] for i in val_indices]
        )
        
        test_class_src, test_acts, test_emotions = self.prepare_classification_data(
            [self.dialogs[i] for i in test_indices],
            [self.act_labels[i] for i in test_indices],
            [self.emotion_labels[i] for i in test_indices]
        )
        
        print(f"{Fore.GREEN}Datos preparados correctamente.{Style.RESET_ALL}")
        print(f"  - Conjunto de entrenamiento: {len(train_src)} ejemplos")
        print(f"  - Conjunto de validación: {len(val_src)} ejemplos")
        print(f"  - Conjunto de prueba: {len(test_src)} ejemplos")
        
        return {
            'train': (train_src, train_trg),
            'val': (val_src, val_trg),
            'test': (test_src, test_trg),
            'train_classification': (train_class_src, train_acts, train_emotions),
            'val_classification': (val_class_src, val_acts, val_emotions),
            'test_classification': (test_class_src, test_acts, test_emotions)
        }
    
    def prepare_generation_data(self, dialogs):
        """
        Prepara datos para generación de respuestas
        Args:
            dialogs: Lista de diálogos
        Returns:
            Tupla de (entradas, objetivos)
        """
        src_sequences = []
        trg_sequences = []
        
        for dialog in dialogs:
            # Para cada par de enunciados consecutivos
            for i in range(len(dialog) - 1):
                src = dialog[i]
                trg = dialog[i + 1]
                
                # Codificar
                src_encoded = self.encode_sequence(src)
                trg_encoded = self.encode_sequence(trg)
                
                # Añadir padding
                src_padded = self.pad_sequence(src_encoded)
                trg_padded = self.pad_sequence(trg_encoded)
                
                src_sequences.append(src_padded)
                trg_sequences.append(trg_padded)
        
        return torch.LongTensor(src_sequences), torch.LongTensor(trg_sequences)
    
    def prepare_classification_data(self, dialogs, act_labels, emotion_labels):
        """
        Prepara datos para clasificación de actos y emociones
        Args:
            dialogs: Lista de diálogos
            act_labels: Lista de etiquetas de actos
            emotion_labels: Lista de etiquetas de emociones
        Returns:
            Tupla de (entradas, etiquetas de actos, etiquetas de emociones)
        """
        src_sequences = []
        act_targets = []
        emotion_targets = []
        
        for dialog, acts, emotions in zip(dialogs, act_labels, emotion_labels):
            for i, (utterance, act, emotion) in enumerate(zip(dialog, acts, emotions)):
                # Codificar
                src_encoded = self.encode_sequence(utterance)
                
                # Añadir padding
                src_padded = self.pad_sequence(src_encoded)
                
                src_sequences.append(src_padded)
                act_targets.append(act)
                emotion_targets.append(emotion)
        
        return torch.LongTensor(src_sequences), torch.LongTensor(act_targets), torch.LongTensor(emotion_targets)
    
    def pad_sequence(self, sequence):
        """
        Añade padding a una secuencia
        Args:
            sequence: Secuencia a rellenar
        Returns:
            Secuencia con padding
        """
        if len(sequence) >= self.max_seq_length:
            return sequence[:self.max_seq_length]
        else:
            return sequence + [self.word2idx['<PAD>']] * (self.max_seq_length - len(sequence))

# Clases de Dataset para PyTorch
class DialogGenerationDataset(torch.utils.data.Dataset):
    def __init__(self, src_sequences, trg_sequences):
        """
        Dataset para generación de diálogos
        Args:
            src_sequences: Secuencias de entrada
            trg_sequences: Secuencias objetivo
        """
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences
    
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, idx):
        return self.src_sequences[idx], self.trg_sequences[idx]

class DialogClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, src_sequences, act_labels, emotion_labels):
        """
        Dataset para clasificación de diálogos
        Args:
            src_sequences: Secuencias de entrada
            act_labels: Etiquetas de actos
            emotion_labels: Etiquetas de emociones
        """
        self.src_sequences = src_sequences
        self.act_labels = act_labels
        self.emotion_labels = emotion_labels
    
    def __len__(self):
        return len(self.src_sequences)
    
    def __getitem__(self, idx):
        return self.src_sequences[idx], self.act_labels[idx], self.emotion_labels[idx]

# Clase de codificación posicional
class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_seq_length, dropout=0.1):
        """
        Codificación posicional para Transformer
        Args:
            d_model: Dimensión del modelo
            max_seq_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)
        
        # Crear matriz de codificación posicional
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Registrar buffer (no es un parámetro pero es parte del módulo)
        self.register_buffer('pe', pe)
    
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from colorama import Fore, Style, init

# Inicializar colorama para colores en la terminal
init(autoreset=True)

# Verificar si CUDA está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        Capa de atención multi-cabeza
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            dropout: Tasa de dropout
        """
        super(MultiHeadAttentionLayer, self).__init__()
        
        assert d_model % num_heads == 0, "d_model debe ser divisible por num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Capas lineales para Q, K, V
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        
        # Capa lineal final
        self.fc_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Factor de escala
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
    
    def forward(self, query, key, value, mask=None):
        """
        Forward pass
        Args:
            query: Tensor de consulta [batch_size, query_len, d_model]
            key: Tensor de clave [batch_size, key_len, d_model]
            value: Tensor de valor [batch_size, value_len, d_model]
            mask: Máscara opcional [batch_size, 1, 1, key_len]
        Returns:
            Tupla de (salida, mapa de atención)
        """
        batch_size = query.shape[0]
        
        # Transformar con capas lineales
        Q = self.fc_q(query)  # [batch_size, query_len, d_model]
        K = self.fc_k(key)    # [batch_size, key_len, d_model]
        V = self.fc_v(value)  # [batch_size, value_len, d_model]
        
        # Reshape para multi-cabeza
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Calcular puntuaciones de atención
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        # Aplicar máscara si se proporciona
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        # Aplicar softmax
        attention = torch.softmax(energy, dim=-1)
        
        # Aplicar dropout
        attention = self.dropout(attention)
        
        # Calcular salida
        x = torch.matmul(attention, V)
        
        # Reshape de vuelta
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.d_model)
        
        # Capa lineal final
        x = self.fc_o(x)
        
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Capa feed-forward por posición
        Args:
            d_model: Dimensión del modelo
            d_ff: Dimensión de la capa feed-forward
            dropout: Tasa de dropout
        """
        super(PositionwiseFeedforwardLayer, self).__init__()
        
        self.fc_1 = nn.Linear(d_model, d_ff)
        self.fc_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Tensor de entrada [batch_size, seq_len, d_model]
        Returns:
            Tensor de salida [batch_size, seq_len, d_model]
        """
        x = self.dropout(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Capa del encoder
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            d_ff: Dimensión de la capa feed-forward
            dropout: Tasa de dropout
        """
        super(EncoderLayer, self).__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.feedforward = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, src, src_mask):
        """
        Forward pass
        Args:
            src: Tensor de entrada [batch_size, src_len, d_model]
            src_mask: Máscara para src [batch_size, 1, 1, src_len]
        Returns:
            Tupla de (salida, mapa de atención)
        """
        # Self attention
        _src, attention = self.self_attention(src, src, src, src_mask)
        
        # Residual connection y layer norm
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        
        # Feedforward
        _src = self.feedforward(src)
        
        # Residual connection y layer norm
        src = self.ff_layer_norm(src + self.dropout(_src))
        
        return src, attention

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.1):
        """
        Encoder completo
        Args:
            input_dim: Dimensión de entrada (tamaño del vocabulario)
            d_model: Dimensión del modelo
            num_layers: Número de capas
            num_heads: Número de cabezas de atención
            d_ff: Dimensión de la capa feed-forward
            max_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(Encoder, self).__init__()
        
        self.tok_embedding = nn.Embedding(input_dim, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, src, src_mask):
        """
        Forward pass
        Args:
            src: Tensor de entrada [batch_size, src_len]
            src_mask: Máscara para src [batch_size, 1, 1, src_len]
        Returns:
            Tupla de (salida, mapas de atención)
        """
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        # Crear posiciones
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        
        # Embedding de tokens y posiciones
        src = self.tok_embedding(src) * self.scale
        src = src + self.pos_embedding(pos)
        src = self.dropout(src)
        
        # Lista para almacenar mapas de atención
        attentions = []
        
        # Pasar por capas
        for layer in self.layers:
            src, attention = layer(src, src_mask)
            attentions.append(attention)
        
        return src, attentions

class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Capa del decoder
        Args:
            d_model: Dimensión del modelo
            num_heads: Número de cabezas de atención
            d_ff: Dimensión de la capa feed-forward
            dropout: Tasa de dropout
        """
        super(DecoderLayer, self).__init__()
        
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        self.enc_attn_layer_norm = nn.LayerNorm(d_model)
        self.ff_layer_norm = nn.LayerNorm(d_model)
        self.self_attention = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.encoder_attention = MultiHeadAttentionLayer(d_model, num_heads, dropout)
        self.feedforward = PositionwiseFeedforwardLayer(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        Forward pass
        Args:
            trg: Tensor objetivo [batch_size, trg_len, d_model]
            enc_src: Salida del encoder [batch_size, src_len, d_model]
            trg_mask: Máscara para trg [batch_size, 1, trg_len, trg_len]
            src_mask: Máscara para src [batch_size, 1, 1, src_len]
        Returns:
            Tupla de (salida, mapa de auto-atención, mapa de atención encoder-decoder)
        """
        # Self attention
        _trg, self_attention = self.self_attention(trg, trg, trg, trg_mask)
        
        # Residual connection y layer norm
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        
        # Encoder attention
        _trg, encoder_attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        
        # Residual connection y layer norm
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        
        # Feedforward
        _trg = self.feedforward(trg)
        
        # Residual connection y layer norm
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        
        return trg, self_attention, encoder_attention

class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_layers, num_heads, d_ff, max_length, dropout=0.1):
        """
        Decoder completo
        Args:
            output_dim: Dimensión de salida (tamaño del vocabulario)
            d_model: Dimensión del modelo
            num_layers: Número de capas
            num_heads: Número de cabezas de atención
            d_ff: Dimensión de la capa feed-forward
            max_length: Longitud máxima de secuencia
            dropout: Tasa de dropout
        """
        super(Decoder, self).__init__()
        
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = nn.Embedding(max_length, d_model)
        
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
    
    def forward(self, trg, enc_src, trg_mask, src_mask):
        """
        Forward pass
        Args:
            trg: Tensor objetivo [batch_size, trg_len]
            enc_src: Salida del encoder [batch_size, src_len, d_model]
            trg_mask: Máscara para trg [batch_size, 1, trg_len, trg_len]
            src_mask: Máscara para src [batch_size, 1, 1, src_len]
        Returns:
            Tupla de (salida, mapas de auto-atención, mapas de atención encoder-decoder)
        """
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        # Crear posiciones
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(device)
        
        # Embedding de tokens y posiciones
        trg = self.tok_embedding(trg) * self.scale
        trg = trg + self.pos_embedding(pos)
        trg = self.dropout(trg)
        
        # Listas para almacenar mapas de atención
        self_attentions = []
        encoder_attentions = []
        
        # Pasar por capas
        for layer in self.layers:
            trg, self_attention, encoder_attention = layer(trg, enc_src, trg_mask, src_mask)
            self_attentions.append(self_attention)
            encoder_attentions.append(encoder_attention)
        
        # Capa lineal final
        output = self.fc_out(trg)
        
        return output, self_attentions, encoder_attentions