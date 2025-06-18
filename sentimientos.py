#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Análisis Comparativo de Modelos RNN/LSTM y Transformer para Procesamiento de Lenguaje Natural
Implementado con PyTorch y soporte para CUDA
"""

import os
import re
import zipfile
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Verificar disponibilidad de CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Utilizando dispositivo: {device}")

# Configuración general
RANDOM_SEED = 42
MAX_VOCAB_SIZE = 15000
MAX_SEQ_LENGTH = 100
BATCH_SIZE = 32  # Reducido para evitar problemas de memoria
EMBEDDING_DIM = 128  # Reducido para evitar problemas de memoria
HIDDEN_DIM = 64  # Reducido para evitar problemas de memoria
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.3

# Configurar semillas para reproducibilidad
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Función para extraer y cargar datos del archivo ZIP
def extraer_cargar_datos(ruta_zip):
    """
    Extrae y carga los datos del archivo ZIP
    
    Args:
        ruta_zip: Ruta al archivo ZIP
        
    Returns:
        DataFrames con los datos
    """
    print(f"Extrayendo archivos de {ruta_zip}...")
    
    # Crear directorio temporal
    temp_dir = tempfile.mkdtemp()
    print(f"Extrayendo archivos a directorio temporal: {temp_dir}")
    
    try:
        # Extraer archivos
        with zipfile.ZipFile(ruta_zip, 'r') as zip_ref:
            # Listar contenido
            contenido = zip_ref.namelist()
            print(f"Archivos en el ZIP: {', '.join(contenido)}")
            
            # Extraer archivos CSV
            archivos_csv = [f for f in contenido if f.endswith('.csv')]
            for archivo in archivos_csv:
                zip_ref.extract(archivo, temp_dir)
                print(f"Extraído: {archivo}")
        
        # Cargar datos
        train_df = pd.read_csv(os.path.join(temp_dir, 'train.csv'))
        val_df = pd.read_csv(os.path.join(temp_dir, 'validation.csv'))
        test_df = pd.read_csv(os.path.join(temp_dir, 'test.csv'))
        
        print(f"Datos cargados correctamente:")
        print(f"- Train: {train_df.shape[0]} filas, {train_df.shape[1]} columnas")
        print(f"- Validation: {val_df.shape[0]} filas, {val_df.shape[1]} columnas")
        print(f"- Test: {test_df.shape[0]} filas, {test_df.shape[1]} columnas")
        
        return {
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df,
            'temp_dir': temp_dir
        }
    
    except Exception as e:
        print(f"Error al extraer o cargar datos: {str(e)}")
        return None

# Función para limpiar recursos
def limpiar_recursos(temp_dir):
    """Elimina el directorio temporal y libera recursos"""
    try:
        import shutil
        shutil.rmtree(temp_dir)
        print(f"Directorio temporal eliminado: {temp_dir}")
    except Exception as e:
        print(f"Error al eliminar directorio temporal: {str(e)}")

# Función para analizar datos
def analizar_datos(train_df, val_df, test_df):
    """
    Analiza los datos y muestra estadísticas
    
    Args:
        train_df: DataFrame de entrenamiento
        val_df: DataFrame de validación
        test_df: DataFrame de prueba
        
    Returns:
        Estadísticas de los datos
    """
    print("\n=== ANÁLISIS DE DATOS ===")
    
    # Información general
    print("\nInformación general:")
    print(f"Conjunto de entrenamiento: {train_df.shape[0]} muestras")
    print(f"Conjunto de validación: {val_df.shape[0]} muestras")
    print(f"Conjunto de prueba: {test_df.shape[0]} muestras")
    
    # Columnas disponibles
    print("\nColumnas disponibles:")
    print(", ".join(train_df.columns))
    
    # Estadísticas de longitud de diálogos
    train_dialog_lengths = train_df['dialog'].str.len()
    
    print("\nEstadísticas de longitud de diálogos (entrenamiento):")
    print(f"Media: {train_dialog_lengths.mean():.2f} caracteres")
    print(f"Mediana: {train_dialog_lengths.median():.2f} caracteres")
    print(f"Mínimo: {train_dialog_lengths.min()} caracteres")
    print(f"Máximo: {train_dialog_lengths.max()} caracteres")
    
    # Distribución de etiquetas (act)
    print("\nDistribución de etiquetas (act) en entrenamiento:")
    
    # Función para extraer primera etiqueta
    def extract_first_label(label_str):
        try:
            if isinstance(label_str, str):
                label_str = label_str.replace('[', '').replace(']', '')
                labels = [int(l.strip()) for l in label_str.split() if l.strip()]
                return labels[0] if labels else 0
            else:
                return 0
        except:
            return 0
    
    train_labels = [extract_first_label(label) for label in train_df['act']]
    unique_labels, counts = np.unique(train_labels, return_counts=True)
    
    for label, count in zip(unique_labels, counts):
        print(f"Etiqueta {label}: {count} muestras ({count/len(train_labels)*100:.2f}%)")
    
    # Gráfico de distribución de etiquetas
    plt.figure(figsize=(10, 6))
    plt.bar(unique_labels, counts)
    plt.title('Distribución de etiquetas (act)')
    plt.xlabel('Etiqueta')
    plt.ylabel('Cantidad')
    plt.savefig('label_distribution.png')
    plt.close()
    
    print(f"Gráfico de distribución guardado como 'label_distribution.png'")
    
    return {
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'num_classes': len(unique_labels)
    }

# Clase para preprocesamiento de texto
class TextPreprocessor:
    def __init__(self, max_vocab_size=MAX_VOCAB_SIZE):
        self.max_vocab_size = max_vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}
        self.word_counts = {}
        self.vocab_size = 2  # PAD y UNK
    
    def clean_text(self, text):
        """Limpia el texto eliminando caracteres especiales"""
        if not isinstance(text, str):
            return ""
        # Convertir a minúsculas
        text = text.lower()
        # Eliminar caracteres especiales y mantener solo letras, números y espacios
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Eliminar espacios múltiples
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def fit(self, texts):
        """Construye el vocabulario a partir de los textos"""
        # Contar frecuencia de palabras
        for text in texts:
            clean_text = self.clean_text(text)
            for word in clean_text.split():
                self.word_counts[word] = self.word_counts.get(word, 0) + 1
        
        # Ordenar palabras por frecuencia
        sorted_words = sorted(self.word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Limitar vocabulario al tamaño máximo
        for word, _ in sorted_words[:self.max_vocab_size - 2]:  # -2 por PAD y UNK
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1
        
        print(f"Vocabulario construido con {self.vocab_size} palabras")
    
    def text_to_sequence(self, text, max_length=None):
        """Convierte texto a secuencia de índices"""
        clean_text = self.clean_text(text)
        sequence = [self.word2idx.get(word, 1) for word in clean_text.split()]  # 1 es <UNK>
        
        # Truncar si es necesario
        if max_length and len(sequence) > max_length:
            sequence = sequence[:max_length]
        
        # Asegurar que la secuencia no esté vacía
        if not sequence:
            sequence = [1]  # Usar <UNK> si la secuencia está vacía
        
        return sequence

# Clase para el conjunto de datos
class DialogDataset(Dataset):
    def __init__(self, texts, labels, preprocessor, max_length=MAX_SEQ_LENGTH):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convertir texto a secuencia
        sequence = self.preprocessor.text_to_sequence(text, self.max_length)
        
        # Convertir a tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return sequence_tensor, label_tensor

# Función para crear lotes con padding
def collate_fn(batch):
    texts, labels = zip(*batch)
    
    # Aplicar padding
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    
    # Convertir etiquetas a tensor
    labels = torch.stack(labels)
    
    return padded_texts, labels

# Modelo LSTM
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout_rate, num_layers=1, bidirectional=True):
        super(LSTMModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Factor 2 si es bidireccional
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc1 = nn.Linear(lstm_output_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch size, sequence length]
        
        embedded = self.embedding(text)
        # embedded shape: [batch size, sequence length, embedding dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: [batch size, sequence length, hidden dim * num directions]
        # hidden shape: [num layers * num directions, batch size, hidden dim]
        
        # Concatenar las salidas finales de ambas direcciones
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]
        # hidden shape: [batch size, hidden dim * num directions]
        
        dense1 = torch.relu(self.fc1(hidden))
        dense1 = self.dropout(dense1)
        # dense1 shape: [batch size, hidden dim]
        
        output = self.fc2(dense1)
        # output shape: [batch size, output dim]
        
        return output

# Modelo Transformer simplificado
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_heads, ff_dim, output_dim, dropout_rate, num_layers=2):
        super(TransformerModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout_rate)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout_rate,
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.fc1 = nn.Linear(embedding_dim, ff_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(ff_dim, output_dim)
        
    def forward(self, text):
        # text shape: [batch size, sequence length]
        
        # Crear máscara para padding
        mask = (text == 0).to(device)
        
        embedded = self.embedding(text)
        # embedded shape: [batch size, sequence length, embedding dim]
        
        embedded = self.pos_encoder(embedded)
        
        # Aplicar transformer
        transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        # transformer_output shape: [batch size, sequence length, embedding dim]
        
        # Pooling global (promedio) - ignorando padding
        mask_expanded = mask.unsqueeze(-1).expand(transformer_output.size())
        transformer_output = transformer_output.masked_fill(mask_expanded, 0.0)
        
        # Suma y promedio solo sobre elementos no enmascarados
        sum_embeddings = transformer_output.sum(dim=1)
        # Contar tokens no enmascarados por secuencia
        mask_sum = (~mask).sum(dim=1).unsqueeze(-1).float()
        # Evitar división por cero
        mask_sum = torch.clamp(mask_sum, min=1.0)
        
        pooled = sum_embeddings / mask_sum
        # pooled shape: [batch size, embedding dim]
        
        dense1 = torch.relu(self.fc1(pooled))
        dense1 = self.dropout(dense1)
        # dense1 shape: [batch size, ff_dim]
        
        output = self.fc2(dense1)
        # output shape: [batch size, output dim]
        
        return output

# Codificación posicional para Transformer
class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Crear matriz de codificación posicional
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        # Registrar buffer (parámetro que no se actualiza durante el entrenamiento)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch size, sequence length, embedding dim]
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

# Función para entrenar modelo
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, model_name):
    """
    Entrena el modelo y devuelve historial de entrenamiento
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        criterion: Función de pérdida
        optimizer: Optimizador
        num_epochs: Número de épocas
        model_name: Nombre del modelo para logs
        
    Returns:
        Historial de entrenamiento
    """
    print(f"\n=== ENTRENAMIENTO DEL MODELO {model_name} ===")
    
    # Mover modelo a GPU si está disponible
    model = model.to(device)
    
    # Historial de entrenamiento
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    # Para early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Modo entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (texts, labels) in enumerate(train_loader):
            # Mover datos a GPU si está disponible
            texts, labels = texts.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            
            try:
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                # Backward pass y optimización
                loss.backward()
                
                # Gradient clipping para evitar explosión de gradientes
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                # Estadísticas
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Mostrar progreso
                if (batch_idx + 1) % 20 == 0:
                    print(f"Época {epoch+1}/{num_epochs}, Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            except RuntimeError as e:
                print(f"Error en batch {batch_idx}: {str(e)}")
                # Intentar liberar memoria
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
        
        # Calcular métricas de entrenamiento
        train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Modo evaluación
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                try:
                    texts, labels = texts.to(device), labels.to(device)
                    outputs = model(texts)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
                except RuntimeError as e:
                    print(f"Error en validación: {str(e)}")
                    continue
        
        # Calcular métricas de validación
        val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Guardar historial
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Mostrar métricas
        print(f"Época {epoch+1}/{num_epochs}, "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Guardar mejor modelo
            torch.save(model.state_dict(), f"{model_name}_best.pt")
            print(f"Modelo guardado como {model_name}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping en época {epoch+1}")
                break
    
    # Cargar mejor modelo
    try:
        model.load_state_dict(torch.load(f"{model_name}_best.pt"))
    except:
        print(f"No se pudo cargar el mejor modelo. Usando el modelo actual.")
    
    return history, model

# Función para evaluar modelo
def evaluate_model(model, test_loader, criterion, num_classes, model_name):
    """
    Evalúa el modelo en el conjunto de prueba
    
    Args:
        model: Modelo a evaluar
        test_loader: DataLoader de prueba
        criterion: Función de pérdida
        num_classes: Número de clases
        model_name: Nombre del modelo para logs
        
    Returns:
        Métricas de evaluación
    """
    print(f"\n=== EVALUACIÓN DEL MODELO {model_name} ===")
    
    # Mover modelo a GPU si está disponible
    model = model.to(device)
    
    # Modo evaluación
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    # Para métricas detalladas
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for texts, labels in test_loader:
            try:
                texts, labels = texts.to(device), labels.to(device)
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                # Guardar para métricas detalladas
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
            except RuntimeError as e:
                print(f"Error en evaluación: {str(e)}")
                continue
    
    # Calcular métricas
    test_loss = test_loss / len(test_loader) if len(test_loader) > 0 else float('inf')
    test_acc = test_correct / test_total if test_total > 0 else 0
    
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # Métricas detalladas
    if len(all_labels) > 0 and len(all_predictions) > 0:
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_predictions, average='weighted', zero_division=0
            )
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Reporte de clasificación
            print("\nReporte de clasificación:")
            print(classification_report(all_labels, all_predictions, zero_division=0))
            
            # Matriz de confusión
            cm = confusion_matrix(all_labels, all_predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Matriz de confusión - {model_name}')
            plt.xlabel('Predicción')
            plt.ylabel('Valor real')
            plt.savefig(f'{model_name}_confusion_matrix.png')
            plt.close()
            
            print(f"Matriz de confusión guardada como '{model_name}_confusion_matrix.png'")
        except Exception as e:
            print(f"Error al calcular métricas detalladas: {str(e)}")
            precision, recall, f1 = 0, 0, 0
    else:
        print("No hay suficientes datos para calcular métricas detalladas")
        precision, recall, f1 = 0, 0, 0
    
    return {
        'test_loss': test_loss,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'labels': all_labels,
        'predictions': all_predictions
    }

# Función para graficar historial de entrenamiento
def plot_training_history(lstm_history, transformer_history):
    """
    Grafica el historial de entrenamiento de ambos modelos
    
    Args:
        lstm_history: Historial del modelo LSTM
        transformer_history: Historial del modelo Transformer
    """
    print("\n=== GRÁFICOS DE ENTRENAMIENTO ===")
    
    # Gráfico de pérdida
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(lstm_history['train_loss'], label='LSTM - Train')
    plt.plot(lstm_history['val_loss'], label='LSTM - Val')
    plt.plot(transformer_history['train_loss'], label='Transformer - Train')
    plt.plot(transformer_history['val_loss'], label='Transformer - Val')
    plt.title('Pérdida durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    # Gráfico de precisión
    plt.subplot(1, 2, 2)
    plt.plot(lstm_history['train_acc'], label='LSTM - Train')
    plt.plot(lstm_history['val_acc'], label='LSTM - Val')
    plt.plot(transformer_history['train_acc'], label='Transformer - Train')
    plt.plot(transformer_history['val_acc'], label='Transformer - Val')
    plt.title('Precisión durante entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("Gráficos de entrenamiento guardados como 'training_history.png'")

# Función para comparar modelos
def compare_models(lstm_results, transformer_results):
    """
    Compara los resultados de ambos modelos
    
    Args:
        lstm_results: Resultados del modelo LSTM
        transformer_results: Resultados del modelo Transformer
    """
    print("\n=== COMPARACIÓN DE MODELOS ===")
    
    # Crear DataFrame para comparación
    comparison = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'LSTM': [
            lstm_results['test_acc'],
            lstm_results['precision'],
            lstm_results['recall'],
            lstm_results['f1']
        ],
        'Transformer': [
            transformer_results['test_acc'],
            transformer_results['precision'],
            transformer_results['recall'],
            transformer_results['f1']
        ]
    })
    
    # Mostrar tabla de comparación
    print("\nComparación de métricas:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    
    # Gráfico de comparación
    plt.figure(figsize=(10, 6))
    
    x = np.arange(len(comparison['Métrica']))
    width = 0.35
    
    plt.bar(x - width/2, comparison['LSTM'], width, label='LSTM')
    plt.bar(x + width/2, comparison['Transformer'], width, label='Transformer')
    
    plt.xlabel('Métrica')
    plt.ylabel('Valor')
    plt.title('Comparación de modelos LSTM vs Transformer')
    plt.xticks(x, comparison['Métrica'])
    plt.xlabel('Métrica')
    plt.ylabel('Valor')
    plt.title('Comparación de modelos LSTM vs Transformer')
    plt.xticks(x, comparison['Métrica'])
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    print("Gráfico de comparación guardado como 'model_comparison.png'")
    
    # Determinar mejor modelo
    lstm_avg = np.mean([
        lstm_results['test_acc'],
        lstm_results['precision'],
        lstm_results['recall'],
        lstm_results['f1']
    ])
    
    transformer_avg = np.mean([
        transformer_results['test_acc'],
        transformer_results['precision'],
        transformer_results['recall'],
        transformer_results['f1']
    ])
    
    mejor_modelo = "LSTM" if lstm_avg > transformer_avg else "Transformer"
    
    print(f"\nModelo con mejor rendimiento general: {mejor_modelo}")
    print(f"Diferencia promedio: {abs(lstm_avg - transformer_avg):.4f}")

# Función para analizar ejemplos específicos
def analyze_examples(test_texts, lstm_model, transformer_model, preprocessor, num_examples=5):
    """
    Analiza ejemplos específicos y compara las predicciones de ambos modelos
    
    Args:
        test_texts: Textos de prueba
        lstm_model: Modelo LSTM
        transformer_model: Modelo Transformer
        preprocessor: Preprocesador de texto
        num_examples: Número de ejemplos a analizar
    """
    print("\n=== ANÁLISIS DE EJEMPLOS ESPECÍFICOS ===")
    
    # Mover modelos a GPU si está disponible
    lstm_model = lstm_model.to(device)
    transformer_model = transformer_model.to(device)
    
    # Modo evaluación
    lstm_model.eval()
    transformer_model.eval()
    
    # Seleccionar ejemplos aleatorios
    if len(test_texts) > 0:
        num_examples = min(num_examples, len(test_texts))
        indices = np.random.choice(len(test_texts), num_examples, replace=False)
        
        for i, idx in enumerate(indices):
            text = test_texts[idx]
            print(f"\nEjemplo {i+1}:")
            print(f"Texto: {text}")
            
            try:
                # Preprocesar texto
                sequence = preprocessor.text_to_sequence(text, MAX_SEQ_LENGTH)
                sequence_tensor = torch.tensor([sequence], dtype=torch.long).to(device)
                
                # Predicciones
                with torch.no_grad():
                    lstm_output = lstm_model(sequence_tensor)
                    transformer_output = transformer_model(sequence_tensor)
                    
                    lstm_pred = torch.argmax(lstm_output, dim=1).item()
                    transformer_pred = torch.argmax(transformer_output, dim=1).item()
                    
                    lstm_probs = torch.softmax(lstm_output, dim=1)[0]
                    transformer_probs = torch.softmax(transformer_output, dim=1)[0]
                
                print(f"Predicción LSTM: Clase {lstm_pred} (Confianza: {lstm_probs[lstm_pred]:.4f})")
                print(f"Predicción Transformer: Clase {transformer_pred} (Confianza: {transformer_probs[transformer_pred]:.4f})")
                
                # Comparación
                if lstm_pred == transformer_pred:
                    print("Ambos modelos coinciden en la predicción.")
                else:
                    print("Los modelos difieren en la predicción.")
                    
                    # Mostrar top-3 probabilidades para cada modelo
                    print("\nTop-3 probabilidades LSTM:")
                    top_lstm = torch.topk(lstm_probs, min(3, len(lstm_probs)))
                    for j in range(min(3, len(top_lstm.indices))):
                        print(f"  Clase {top_lstm.indices[j].item()}: {top_lstm.values[j].item():.4f}")
                        
                    print("\nTop-3 probabilidades Transformer:")
                    top_transformer = torch.topk(transformer_probs, min(3, len(transformer_probs)))
                    for j in range(min(3, len(top_transformer.indices))):
                        print(f"  Clase {top_transformer.indices[j].item()}: {top_transformer.values[j].item():.4f}")
            except Exception as e:
                print(f"Error al analizar ejemplo: {str(e)}")
    else:
        print("No hay ejemplos disponibles para analizar")

# Función para analizar hiperparámetros
def analyze_hyperparameters(train_data, train_labels, val_data, val_labels, preprocessor, num_classes):
    """
    Analiza el impacto de diferentes hiperparámetros en el rendimiento de los modelos
    
    Args:
        train_data: Datos de entrenamiento
        train_labels: Etiquetas de entrenamiento
        val_data: Datos de validación
        val_labels: Etiquetas de validación
        preprocessor: Preprocesador de texto
        num_classes: Número de clases
    """
    print("\n=== ANÁLISIS DE HIPERPARÁMETROS ===")
    
    # Crear conjuntos de datos
    train_dataset = DialogDataset(train_data, train_labels, preprocessor)
    val_dataset = DialogDataset(val_data, val_labels, preprocessor)
    
    # Parámetros a probar para LSTM - reducidos para evitar problemas de memoria
    lstm_params = [
        {'learning_rate': 0.001, 'batch_size': 16, 'hidden_dim': 32},
        {'learning_rate': 0.0005, 'batch_size': 16, 'hidden_dim': 64}
    ]
    
    # Parámetros a probar para Transformer - reducidos para evitar problemas de memoria
    transformer_params = [
        {'learning_rate': 0.0005, 'batch_size': 16, 'num_heads': 2},
        {'learning_rate': 0.0001, 'batch_size': 16, 'num_heads': 4}
    ]
    
    # Resultados
    lstm_results = []
    transformer_results = []
    
    # Probar hiperparámetros para LSTM
    print("\n--- Análisis de hiperparámetros para LSTM ---")
    
    for i, params in enumerate(lstm_params):
        print(f"\nPrueba {i+1}/{len(lstm_params)} - LSTM")
        print(f"Parámetros: {params}")
        
        # Crear DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params['batch_size'], 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Crear modelo
        model = LSTMModel(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=params['hidden_dim'],
            output_dim=num_classes,
            dropout_rate=DROPOUT_RATE
        )
        
        # Criterio y optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        try:
            # Entrenar modelo (con menos épocas para pruebas rápidas)
            history, model = train_model(
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                num_epochs=3,  # Menos épocas para pruebas rápidas
                model_name=f"lstm_test_{i+1}"
            )
            
            # Evaluar modelo
            val_loss = min(history['val_loss'])
            val_acc = max(history['val_acc'])
            
            # Guardar resultados
            lstm_results.append({
                'params': params,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            print(f"Mejor pérdida en validación: {val_loss:.4f}")
            print(f"Mejor precisión en validación: {val_acc:.4f}")
        except Exception as e:
            print(f"Error en prueba de hiperparámetros LSTM: {str(e)}")
            # Agregar resultado con valores por defecto
            lstm_results.append({
                'params': params,
                'val_loss': float('inf'),
                'val_acc': 0.0
            })
    
    # Probar hiperparámetros para Transformer
    print("\n--- Análisis de hiperparámetros para Transformer ---")
    
    for i, params in enumerate(transformer_params):
        print(f"\nPrueba {i+1}/{len(transformer_params)} - Transformer")
        print(f"Parámetros: {params}")
        
        # Crear DataLoader
        train_loader = DataLoader(
            train_dataset, 
            batch_size=params['batch_size'], 
            shuffle=True, 
            collate_fn=collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=params['batch_size'], 
            shuffle=False, 
            collate_fn=collate_fn
        )
        
        # Crear modelo
        model = TransformerModel(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            num_heads=params['num_heads'],
            ff_dim=128,  # Reducido para evitar problemas de memoria
            output_dim=num_classes,
            dropout_rate=DROPOUT_RATE
        )
        
        # Criterio y optimizador
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        try:
            # Entrenar modelo (con menos épocas para pruebas rápidas)
            history, model = train_model(
                model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                num_epochs=3,  # Menos épocas para pruebas rápidas
                model_name=f"transformer_test_{i+1}"
            )
            
            # Evaluar modelo
            val_loss = min(history['val_loss'])
            val_acc = max(history['val_acc'])
            
            # Guardar resultados
            transformer_results.append({
                'params': params,
                'val_loss': val_loss,
                'val_acc': val_acc
            })
            
            print(f"Mejor pérdida en validación: {val_loss:.4f}")
            print(f"Mejor precisión en validación: {val_acc:.4f}")
        except Exception as e:
            print(f"Error en prueba de hiperparámetros Transformer: {str(e)}")
            # Agregar resultado con valores por defecto
            transformer_results.append({
                'params': params,
                'val_loss': float('inf'),
                'val_acc': 0.0
            })
    
    # Mostrar resultados
    print("\n--- Resultados del análisis de hiperparámetros ---")
    
    # Mejores hiperparámetros para LSTM
    if lstm_results:
        best_lstm = min(lstm_results, key=lambda x: x['val_loss'])
        print("\nMejores hiperparámetros para LSTM:")
        for k, v in best_lstm['params'].items():
            print(f"- {k}: {v}")
        print(f"Precisión en validación: {best_lstm['val_acc']:.4f}")
    else:
        print("\nNo hay resultados disponibles para LSTM")
        best_lstm = {'params': {'learning_rate': 0.001, 'batch_size': 16, 'hidden_dim': 64}}
    
    # Mejores hiperparámetros para Transformer
    if transformer_results:
        best_transformer = min(transformer_results, key=lambda x: x['val_loss'])
        print("\nMejores hiperparámetros para Transformer:")
        for k, v in best_transformer['params'].items():
            print(f"- {k}: {v}")
        print(f"Precisión en validación: {best_transformer['val_acc']:.4f}")
    else:
        print("\nNo hay resultados disponibles para Transformer")
        best_transformer = {'params': {'learning_rate': 0.0001, 'batch_size': 16, 'num_heads': 2}}
    
    # Gráficos de comparación si hay resultados
    if lstm_results and transformer_results:
        plt.figure(figsize=(15, 6))
        
        # LSTM
        plt.subplot(1, 2, 1)
        plt.bar(
            range(len(lstm_results)),
            [r['val_acc'] for r in lstm_results],
            color='skyblue'
        )
        plt.title('Precisión en validación - LSTM')
        plt.xlabel('Configuración')
        plt.ylabel('Precisión')
        plt.xticks(
            range(len(lstm_results)),
            [f"LR={r['params']['learning_rate']}\nBS={r['params']['batch_size']}\nHD={r['params']['hidden_dim']}" 
             for r in lstm_results],
            rotation=45
        )
        
        # Transformer
        plt.subplot(1, 2, 2)
        plt.bar(
            range(len(transformer_results)),
            [r['val_acc'] for r in transformer_results],
            color='salmon'
        )
        plt.title('Precisión en validación - Transformer')
        plt.xlabel('Configuración')
        plt.ylabel('Precisión')
        plt.xticks(
            range(len(transformer_results)),
            [f"LR={r['params']['learning_rate']}\nBS={r['params']['batch_size']}\nNH={r['params']['num_heads']}" 
             for r in transformer_results],
            rotation=45
        )
        
        plt.tight_layout()
        plt.savefig('hyperparameter_analysis.png')
        plt.close()
        
        print("Gráfico de análisis de hiperparámetros guardado como 'hyperparameter_analysis.png'")
    
    return {
        'lstm_best': best_lstm,
        'transformer_best': best_transformer
    }

# Función principal
def main():
    """Función principal que ejecuta todo el flujo de trabajo"""
    print("=== ANÁLISIS COMPARATIVO DE MODELOS RNN/LSTM Y TRANSFORMER PARA NLP ===")
    print(f"Utilizando dispositivo: {device}")
    
    # Verificar si existe el archivo ZIP
    zip_path = 'switchboard.zip'
    if not os.path.exists(zip_path):
        print(f"Error: No se encontró el archivo {zip_path}")
        print("Por favor, asegúrate de que el archivo ZIP esté en el directorio actual.")
        return
    
    # 1. Extraer y cargar datos
    datos = extraer_cargar_datos(zip_path)
    if not datos:
        return
    
    # 2. Analizar datos
    datos_analizados = analizar_datos(
        datos['train_df'],
        datos['val_df'],
        datos['test_df']
    )
    
    # 3. Preprocesar datos
    print("\n=== PREPROCESAMIENTO DE DATOS ===")
    
    # Función para extraer primera etiqueta
    def extract_first_label(label_str):
        try:
            if isinstance(label_str, str):
                label_str = label_str.replace('[', '').replace(']', '')
                labels = [int(l.strip()) for l in label_str.split() if l.strip()]
                return labels[0] if labels else 0
            else:
                return 0
        except:
            return 0
    
    # Extraer textos y etiquetas
    train_texts = datos['train_df']['dialog'].tolist()
    train_labels = [extract_first_label(label) for label in datos['train_df']['act']]
    
    val_texts = datos['val_df']['dialog'].tolist()
    val_labels = [extract_first_label(label) for label in datos['val_df']['act']]
    
    test_texts = datos['test_df']['dialog'].tolist()
    test_labels = [extract_first_label(label) for label in datos['test_df']['act']]
    
    # Crear preprocesador
    preprocessor = TextPreprocessor(max_vocab_size=MAX_VOCAB_SIZE)
    preprocessor.fit(train_texts)
    
    # Número de clases
    num_classes = datos_analizados['num_classes']
    print(f"Número de clases: {num_classes}")
    
    # 4. Crear conjuntos de datos
    train_dataset = DialogDataset(train_texts, train_labels, preprocessor, MAX_SEQ_LENGTH)
    val_dataset = DialogDataset(val_texts, val_labels, preprocessor, MAX_SEQ_LENGTH)
    test_dataset = DialogDataset(test_texts, test_labels, preprocessor, MAX_SEQ_LENGTH)
    
    # 5. Crear DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=collate_fn
    )
    
    # 6. Analizar hiperparámetros (versión simplificada para evitar errores)
    try:
        hyperparams = analyze_hyperparameters(
            train_texts[:1000],  # Usar subconjunto para análisis rápido
            train_labels[:1000], 
            val_texts[:200], 
            val_labels[:200], 
            preprocessor, 
            num_classes
        )
    except Exception as e:
        print(f"Error en análisis de hiperparámetros: {str(e)}")
        # Valores por defecto
        hyperparams = {
            'lstm_best': {'params': {'learning_rate': 0.001, 'batch_size': 16, 'hidden_dim': 64}},
            'transformer_best': {'params': {'learning_rate': 0.0001, 'batch_size': 16, 'num_heads': 2}}
        }
    
    # 7. Crear y entrenar modelo LSTM con los mejores hiperparámetros
    try:
        lstm_model = LSTMModel(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=hyperparams['lstm_best']['params']['hidden_dim'],
            output_dim=num_classes,
            dropout_rate=DROPOUT_RATE
        )
        
        # Criterio y optimizador para LSTM
        lstm_criterion = nn.CrossEntropyLoss()
        lstm_optimizer = optim.Adam(
            lstm_model.parameters(), 
            lr=hyperparams['lstm_best']['params']['learning_rate']
        )
        
        # Entrenar modelo LSTM
        lstm_history, lstm_model = train_model(
            lstm_model, 
            train_loader, 
            val_loader, 
            lstm_criterion, 
            lstm_optimizer, 
            num_epochs=5,  # Reducido para evitar problemas
            model_name="lstm_final"
        )
    except Exception as e:
        print(f"Error en entrenamiento de LSTM: {str(e)}")
        # Crear modelo simple para continuar
        lstm_model = LSTMModel(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            hidden_dim=64,
            output_dim=num_classes,
            dropout_rate=DROPOUT_RATE
        ).to(device)
        lstm_history = {'train_loss': [0], 'train_acc': [0], 'val_loss': [0], 'val_acc': [0]}
    
    # 8. Crear y entrenar modelo Transformer con los mejores hiperparámetros
    try:
        transformer_model = TransformerModel(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            num_heads=hyperparams['transformer_best']['params']['num_heads'],
            ff_dim=128,  # Reducido para evitar problemas de memoria
            output_dim=num_classes,
            dropout_rate=DROPOUT_RATE
        )
        
        # Criterio y optimizador para Transformer
        transformer_criterion = nn.CrossEntropyLoss()
        transformer_optimizer = optim.Adam(
            transformer_model.parameters(), 
            lr=hyperparams['transformer_best']['params']['learning_rate']
        )
        
        # Entrenar modelo Transformer
        transformer_history, transformer_model = train_model(
            transformer_model, 
            train_loader, 
            val_loader, 
            transformer_criterion, 
            transformer_optimizer, 
            num_epochs=5,  # Reducido para evitar problemas
            model_name="transformer_final"
        )
    except Exception as e:
        print(f"Error en entrenamiento de Transformer: {str(e)}")
        # Crear modelo simple para continuar
        transformer_model = TransformerModel(
            vocab_size=preprocessor.vocab_size,
            embedding_dim=EMBEDDING_DIM,
            num_heads=2,
            ff_dim=128,
            output_dim=num_classes,
            dropout_rate=DROPOUT_RATE
        ).to(device)
        transformer_history = {'train_loss': [0], 'train_acc': [0], 'val_loss': [0], 'val_acc': [0]}
    
    # 9. Graficar historial de entrenamiento
    try:
        plot_training_history(lstm_history, transformer_history)
    except Exception as e:
        print(f"Error al graficar historial: {str(e)}")
    
    # 10. Evaluar modelos en conjunto de prueba
    try:
        lstm_results = evaluate_model(
            lstm_model, 
            test_loader, 
            lstm_criterion, 
            num_classes, 
            "LSTM"
        )
    except Exception as e:
        print(f"Error al evaluar LSTM: {str(e)}")
        lstm_results = {'test_loss': 0, 'test_acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    try:
        transformer_results = evaluate_model(
            transformer_model, 
            test_loader, 
            transformer_criterion, 
            num_classes, 
            "Transformer"
        )
    except Exception as e:
        print(f"Error al evaluar Transformer: {str(e)}")
        transformer_results = {'test_loss': 0, 'test_acc': 0, 'precision': 0, 'recall': 0, 'f1': 0}
    
    # 11. Comparar modelos
    try:
        compare_models(lstm_results, transformer_results)
    except Exception as e:
        print(f"Error al comparar modelos: {str(e)}")
    
    # 12. Analizar ejemplos específicos
    try:
        analyze_examples(
            test_texts[:100],  # Usar subconjunto para análisis rápido
            lstm_model, 
            transformer_model, 
            preprocessor, 
            num_examples=3
        )
    except Exception as e:
        print(f"Error al analizar ejemplos: {str(e)}")
    
    # 13. Conclusiones
    print("\n=== CONCLUSIONES ===")
    
    # Comparar rendimiento general
    lstm_avg = np.mean([
        lstm_results['test_acc'],
        lstm_results['precision'],
        lstm_results['recall'],
        lstm_results['f1']
    ])
    
    transformer_avg = np.mean([
        transformer_results['test_acc'],
        transformer_results['precision'],
        transformer_results['recall'],
        transformer_results['f1']
    ])
    
    mejor_modelo = "LSTM" if lstm_avg > transformer_avg else "Transformer"
    
    print(f"1. Modelo con mejor rendimiento general: {mejor_modelo}")
    print(f"   - Rendimiento promedio LSTM: {lstm_avg:.4f}")
    print(f"   - Rendimiento promedio Transformer: {transformer_avg:.4f}")
    print(f"   - Diferencia: {abs(lstm_avg - transformer_avg):.4f}")
    
    # Ventajas y desventajas
    print("\n2. Ventajas y desventajas:")
    
    print("\n   LSTM:")
    print("   - Ventajas: Capacidad para capturar dependencias a largo plazo, manejo eficiente de secuencias de longitud variable")
    print("   - Desventajas: Entrenamiento secuencial que puede ser lento, dificultad para paralelizar")
    
    print("\n   Transformer:")
    print("   - Ventajas: Paralelización eficiente, capacidad para capturar relaciones globales mediante mecanismo de atención")
    print("   - Desventajas: Mayor complejidad computacional, requiere más datos para entrenar efectivamente")
    
    # Impacto de hiperparámetros
    print("\n3. Impacto de hiperparámetros:")
    
    print("\n   LSTM:")
    print(f"   - Mejor tasa de aprendizaje: {hyperparams['lstm_best']['params']['learning_rate']}")
    print(f"   - Mejor tamaño de lote: {hyperparams['lstm_best']['params']['batch_size']}")
    print(f"   - Mejor dimensión oculta: {hyperparams['lstm_best']['params']['hidden_dim']}")
    
    print("\n   Transformer:")
    print(f"   - Mejor tasa de aprendizaje: {hyperparams['transformer_best']['params']['learning_rate']}")
    print(f"   - Mejor tamaño de lote: {hyperparams['transformer_best']['params']['batch_size']}")
    print(f"   - Mejor número de cabezas de atención: {hyperparams['transformer_best']['params']['num_heads']}")
    
    # Recomendaciones
    print("\n4. Recomendaciones:")
    print("   - Utilizar LSTM para secuencias cortas o medianas donde la eficiencia computacional es importante")
    print("   - Utilizar Transformer para tareas que requieren capturar relaciones globales en el texto")
    print("   - Experimentar con arquitecturas híbridas que combinen las fortalezas de ambos modelos")
    print("   - Aplicar técnicas de regularización como dropout y early stopping para evitar sobreajuste")
    print("   - Ajustar hiperparámetros como tasa de aprendizaje y tamaño de lote según las características específicas del problema")
    
    # Limpiar recursos
    limpiar_recursos(datos['temp_dir'])
    
    print("\n=== ANÁLISIS COMPLETADO ===")

# Ejecutar función principal si se ejecuta como script
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error en la ejecución principal: {str(e)}")
        # Si hay error de CUDA, intentar liberar memoria
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
