# Implementación de Modelos RNN/LSTM y Transformer para NLP
# Basado en la rúbrica de evaluación proporcionada

import json
import math
import os
import time
import warnings

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from rouge import Rouge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Ignorar advertencias
warnings.filterwarnings("ignore")

# Verificar disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Utilizando dispositivo: {device}")

# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# Configuración de semilla para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Cargar la rúbrica de evaluación
try:
    with open("rubrica_evaluacion.json", "r", encoding="utf-8") as f:
        rubrica = json.load(f)
    print("Rúbrica de evaluación cargada correctamente")
except Exception as e:
    print(f"Error al cargar la rúbrica: {e}")
    rubrica = {
        "rubrica": {
            "metricas_evaluacion": {
                "rnn_lstm": ["accuracy", "precision", "recall", "F1-score"],
                "transformer": ["BLEU Score", "ROUGE"],
            }
        }
    }

# Cargar los datos desde archivos parquet
print("Cargando datos...")
try:
    train_data = pd.read_parquet("train.parquet")
    val_data = pd.read_parquet("validation.parquet")
    test_data = pd.read_parquet("test.parquet")
    print(
        f"Datos cargados: {len(train_data)} ejemplos de entrenamiento, {len(val_data)} de validación, {len(test_data)} de prueba"
    )
except Exception as e:
    print(f"Error al cargar los datos: {e}")
    print("Generando datos sintéticos para demostración...")
    # Generar datos sintéticos para demostración
    from sklearn.datasets import fetch_20newsgroups

    newsgroups = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )

    # Crear DataFrame con textos y etiquetas
    data = pd.DataFrame(
        {"text": newsgroups.data[:1000], "target": newsgroups.target[:1000]}
    )

    # Dividir en train, val, test
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

    print(
        f"Datos sintéticos generados: {len(train_data)} ejemplos de entrenamiento, {len(val_data)} de validación, {len(test_data)} de prueba"
    )

# Mostrar información sobre los datos
print("\nEstructura de los datos de entrenamiento:")
print(train_data.head())
print("\nColumnas disponibles:")
print(train_data.columns.tolist())


# Preprocesamiento de datos
class TextProcessor:
    def __init__(self, max_vocab_size=10000, max_seq_length=100):
        self.max_vocab_size = max_vocab_size
        self.max_seq_length = max_seq_length
        self.word2idx = {"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}
        self.idx2word = {0: "<PAD>", 1: "<UNK>", 2: "<SOS>", 3: "<EOS>"}
        self.word_freq = {}
        self.vocab_size = 4  # Inicialmente tenemos 4 tokens especiales

    def build_vocab(self, texts):
        """Construye el vocabulario a partir de los textos de entrenamiento"""
        # Contar frecuencia de palabras
        for text in texts:
            if isinstance(text, str):  # Asegurarse de que el texto es una cadena
                for word in nltk.word_tokenize(text.lower()):
                    if word not in self.word_freq:
                        self.word_freq[word] = 1
                    else:
                        self.word_freq[word] += 1

        # Ordenar palabras por frecuencia (descendente)
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        # Añadir palabras al vocabulario (limitado por max_vocab_size)
        for word, freq in sorted_words[
            : self.max_vocab_size - 4
        ]:  # -4 por los tokens especiales
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

        print(f"Vocabulario construido con {self.vocab_size} palabras")

    def text_to_indices(self, text, add_special_tokens=False):
        """Convierte un texto en una secuencia de índices"""
        if not isinstance(text, str):
            text = str(text)

        tokens = nltk.word_tokenize(text.lower())
        indices = []

        if add_special_tokens:
            indices.append(self.word2idx["<SOS>"])

        for token in tokens[
            : self.max_seq_length - 2 if add_special_tokens else self.max_seq_length
        ]:
            if token in self.word2idx:
                indices.append(self.word2idx[token])
            else:
                indices.append(self.word2idx["<UNK>"])

        if add_special_tokens:
            indices.append(self.word2idx["<EOS>"])

        # Padding
        if len(indices) < self.max_seq_length:
            indices += [self.word2idx["<PAD>"]] * (self.max_seq_length - len(indices))
        else:
            indices = indices[: self.max_seq_length]

        return indices

    def indices_to_text(self, indices):
        """Convierte una secuencia de índices en texto"""
        tokens = []
        for idx in indices:
            if idx == self.word2idx["<PAD>"] or idx == self.word2idx["<EOS>"]:
                break
            if idx != self.word2idx["<SOS>"]:
                tokens.append(self.idx2word.get(idx, "<UNK>"))
        return " ".join(tokens)


# Preparar los datos
print("Preparando los datos...")

# Determinar las columnas de entrada y salida según la estructura de los datos
# Esto puede necesitar ajustes según tus datos específicos
if "text" in train_data.columns and "target" in train_data.columns:
    input_col = "text"
    output_col = "target"
elif len(train_data.columns) >= 2:
    input_col = train_data.columns[0]
    output_col = train_data.columns[1]
else:
    input_col = train_data.columns[0]
    output_col = train_data.columns[0]  # Usar la misma columna como entrada y salida

print(f"Usando {input_col} como entrada y {output_col} como salida")

# Inicializar el procesador de texto
text_processor = TextProcessor(max_vocab_size=10000, max_seq_length=100)

# Construir vocabulario con los datos de entrenamiento
all_texts = []
for text in train_data[input_col]:
    if isinstance(text, str):
        all_texts.append(text)
    else:
        all_texts.append(str(text))

if input_col != output_col:
    for text in train_data[output_col]:
        if isinstance(text, str):
            all_texts.append(text)
        else:
            all_texts.append(str(text))

text_processor.build_vocab(all_texts)


# Clase de Dataset personalizada para secuencias
class SequenceDataset(Dataset):
    def __init__(self, input_texts, output_texts, text_processor, is_transformer=False):
        self.input_texts = input_texts
        self.output_texts = output_texts
        self.text_processor = text_processor
        self.is_transformer = is_transformer

    def __len__(self):
        return len(self.input_texts)

    def __getitem__(self, idx):
        input_text = self.input_texts[idx]
        output_text = self.output_texts[idx]

        # Convertir textos a secuencias de índices
        input_indices = self.text_processor.text_to_indices(
            input_text, add_special_tokens=True
        )
        output_indices = self.text_processor.text_to_indices(
            output_text, add_special_tokens=True
        )

        # Convertir a tensores
        input_tensor = torch.tensor(input_indices, dtype=torch.long)
        output_tensor = torch.tensor(output_indices, dtype=torch.long)

        if self.is_transformer:
            # Para transformer, necesitamos máscaras de atención
            input_mask = (input_tensor != self.text_processor.word2idx["<PAD>"]).float()
            output_mask = (
                output_tensor != self.text_processor.word2idx["<PAD>"]
            ).float()
            return input_tensor, output_tensor, input_mask, output_mask
        else:
            return input_tensor, output_tensor


# Crear datasets
train_dataset = SequenceDataset(
    train_data[input_col].tolist(), train_data[output_col].tolist(), text_processor
)

val_dataset = SequenceDataset(
    val_data[input_col].tolist(), val_data[output_col].tolist(), text_processor
)

test_dataset = SequenceDataset(
    test_data[input_col].tolist(), test_data[output_col].tolist(), text_processor
)

# Crear dataloaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

print(f"Dataloaders creados con batch_size={batch_size}")


# Definición de modelos
class SimpleRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.rnn = nn.RNN(
            emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch_size, src_len, emb_dim]

        outputs, hidden = self.rnn(embedded)
        # outputs = [batch_size, src_len, hidden_dim]
        # hidden = [n_layers, batch_size, hidden_dim]

        predictions = self.fc_out(outputs)
        # predictions = [batch_size, src_len, output_dim]

        return predictions


class LSTM(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hidden_dim,
        output_dim,
        n_layers,
        dropout,
        bidirectional=False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(
            emb_dim,
            hidden_dim,
            n_layers,
            dropout=dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc_out = nn.Linear(
            hidden_dim * 2 if bidirectional else hidden_dim, output_dim
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch_size, src_len, emb_dim]

        outputs, (hidden, cell) = self.lstm(embedded)
        # outputs = [batch_size, src_len, hidden_dim * n_directions]
        # hidden = [n_layers * n_directions, batch_size, hidden_dim]
        # cell = [n_layers * n_directions, batch_size, hidden_dim]

        predictions = self.fc_out(outputs)
        # predictions = [batch_size, src_len, output_dim]

        return predictions


class GRU(nn.Module):
    def __init__(self, input_dim, emb_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.gru = nn.GRU(
            emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [batch_size, src_len]
        embedded = self.dropout(self.embedding(src))
        # embedded = [batch_size, src_len, emb_dim]

        outputs, hidden = self.gru(embedded)
        # outputs = [batch_size, src_len, hidden_dim]
        # hidden = [n_layers, batch_size, hidden_dim]

        predictions = self.fc_out(outputs)
        # predictions = [batch_size, src_len, output_dim]

        return predictions


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        hidden_dim,
        output_dim,
        n_layers,
        n_heads,
        dropout,
        max_length=100,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim, dropout)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, n_layers)

        self.fc_out = nn.Linear(emb_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src = [batch_size, src_len]
        embedded = self.embedding(src) * math.sqrt(self.embedding.embedding_dim)
        # embedded = [batch_size, src_len, emb_dim]

        embedded = self.pos_encoder(embedded)

        # Corregir la máscara de padding
        if src_mask is None:
            # Crear máscara de padding (1 para tokens reales, 0 para padding)
            src_key_padding_mask = src == 0  # [batch_size, src_len]
        else:
            src_key_padding_mask = src_mask

        outputs = self.transformer_encoder(
            embedded, src_key_padding_mask=src_key_padding_mask
        )
        # outputs = [batch_size, src_len, emb_dim]

        predictions = self.fc_out(outputs)
        # predictions = [batch_size, src_len, output_dim]

        return predictions


# Funciones de entrenamiento y evaluación
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    total_samples = 0

    for batch_idx, (src, trg) in enumerate(tqdm(dataloader, desc="Training")):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        # Forward pass
        output = model(src)

        # Reshape para calcular pérdida
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg.view(-1)

        # Calcular pérdida
        loss = criterion(output, trg)

        # Backward pass
        loss.backward()

        # Actualizar pesos
        optimizer.step()

        # Calcular precisión
        _, predicted = torch.max(output, 1)
        correct = (predicted == trg).float()
        mask = (trg != 0).float()  # Ignorar padding
        correct = (correct * mask).sum().item()
        total = mask.sum().item()

        # Actualizar métricas
        epoch_loss += loss.item() * src.size(0)
        epoch_acc += correct
        total_samples += total

    return epoch_loss / len(dataloader.dataset), epoch_acc / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    total_samples = 0

    all_preds = []
    all_trgs = []

    with torch.no_grad():
        for batch_idx, (src, trg) in enumerate(tqdm(dataloader, desc="Evaluating")):
            src, trg = src.to(device), trg.to(device)

            # Forward pass
            output = model(src)

            # Reshape para calcular pérdida
            output_dim = output.shape[-1]
            output_flat = output.view(-1, output_dim)
            trg_flat = trg.view(-1)

            # Calcular pérdida
            loss = criterion(output_flat, trg_flat)

            # Calcular precisión
            _, predicted = torch.max(output_flat, 1)
            correct = (predicted == trg_flat).float()
            mask = (trg_flat != 0).float()  # Ignorar padding
            correct = (correct * mask).sum().item()
            total = mask.sum().item()

            # Actualizar métricas
            epoch_loss += loss.item() * src.size(0)
            epoch_acc += correct
            total_samples += total

            # Guardar predicciones y targets para calcular métricas adicionales
            for i in range(src.size(0)):
                pred_seq = torch.argmax(output[i], dim=1).cpu().numpy()
                trg_seq = trg[i].cpu().numpy()

                # Filtrar padding
                pred_seq = pred_seq[trg_seq != 0]
                trg_seq = trg_seq[trg_seq != 0]

                all_preds.append(pred_seq)
                all_trgs.append(trg_seq)

    return (
        epoch_loss / len(dataloader.dataset),
        epoch_acc / total_samples,
        all_preds,
        all_trgs,
    )


def calculate_metrics(predictions, targets, idx2word):
    """
    Calcula métricas adicionales como F1, precisión, recall y BLEU/ROUGE
    """
    # Convertir índices a palabras
    pred_texts = []
    target_texts = []

    for pred, target in zip(predictions, targets):
        pred_text = [
            idx2word.get(idx, "<UNK>") for idx in pred if idx > 3
        ]  # Ignorar tokens especiales
        target_text = [
            idx2word.get(idx, "<UNK>") for idx in target if idx > 3
        ]  # Ignorar tokens especiales

        pred_texts.append(pred_text)
        target_texts.append([target_text])  # BLEU espera una lista de referencias

    # Calcular BLEU
    try:
        smoothie = SmoothingFunction().method1
        bleu_score = corpus_bleu(target_texts, pred_texts, smoothing_function=smoothie)
    except:
        bleu_score = 0

    # Calcular ROUGE
    try:
        rouge = Rouge()

        # Convertir listas de tokens a strings
        pred_strings = [" ".join(pred) for pred in pred_texts]
        target_strings = [" ".join(target[0]) for target in target_texts]

        # Asegurarse de que no hay strings vacíos
        valid_pairs = [(p, t) for p, t in zip(pred_strings, target_strings) if p and t]

        if valid_pairs:
            pred_valid, target_valid = zip(*valid_pairs)
            rouge_scores = rouge.get_scores(pred_valid, target_valid, avg=True)
            rouge_1 = rouge_scores["rouge-1"]["f"]
            rouge_2 = rouge_scores["rouge-2"]["f"]
            rouge_l = rouge_scores["rouge-l"]["f"]
        else:
            rouge_1 = rouge_2 = rouge_l = 0
    except:
        rouge_1 = rouge_2 = rouge_l = 0

    # Calcular precisión, recall y F1 (para tareas de clasificación)
    # Aplanar todas las predicciones y targets
    all_preds = []
    all_targets = []

    for pred, target in zip(predictions, targets):
        all_preds.extend(pred)
        all_targets.extend(target)

    try:
        precision = precision_score(
            all_targets, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_targets, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        accuracy = accuracy_score(all_targets, all_preds)
    except:
        precision = recall = f1 = accuracy = 0

    return {
        "bleu": bleu_score,
        "rouge-1": rouge_1,
        "rouge-2": rouge_2,
        "rouge-l": rouge_l,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
    }


def train_model(
    model, train_loader, val_loader, optimizer, criterion, n_epochs, device, model_name
):
    """
    Entrena un modelo y guarda el mejor modelo basado en la pérdida de validación
    """
    best_valid_loss = float("inf")
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    for epoch in range(n_epochs):
        start_time = time.time()

        # Entrenar una época
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluar en conjunto de validación
        valid_loss, valid_acc, _, _ = evaluate(model, val_loader, criterion, device)

        # Guardar métricas
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accs.append(valid_acc)

        # Guardar el mejor modelo
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f"{model_name}_best.pt")

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs:.2f}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\tValid Loss: {valid_loss:.3f} | Valid Acc: {valid_acc*100:.2f}%")

    # Cargar el mejor modelo
    model.load_state_dict(torch.load(f"{model_name}_best.pt"))

    # Devolver historiales para visualización
    history = {
        "train_loss": train_losses,
        "train_acc": train_accs,
        "val_loss": valid_losses,
        "val_acc": valid_accs,
    }

    return model, history


def evaluate_model(model, test_loader, criterion, device, idx2word):
    """
    Evalúa un modelo en el conjunto de prueba y calcula métricas adicionales
    """
    test_loss, test_acc, all_preds, all_trgs = evaluate(
        model, test_loader, criterion, device
    )

    print(f"Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%")

    # Calcular métricas adicionales
    metrics = calculate_metrics(all_preds, all_trgs, idx2word)

    print(f"Métricas adicionales:")
    print(f"BLEU: {metrics['bleu']:.4f}")
    print(f"ROUGE-1: {metrics['rouge-1']:.4f}")
    print(f"ROUGE-2: {metrics['rouge-2']:.4f}")
    print(f"ROUGE-L: {metrics['rouge-l']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1: {metrics['f1']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")

    return metrics


def plot_training_history(history, model_name):
    """
    Visualiza el historial de entrenamiento
    """
    plt.figure(figsize=(12, 5))

    # Gráfico de pérdida
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train")
    plt.plot(history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Gráfico de precisión
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{model_name}_history.png")
    plt.close()


def compare_models(metrics_dict, model_names, metric_names):
    """
    Compara diferentes modelos según varias métricas
    """
    plt.figure(figsize=(15, 10))

    for i, metric in enumerate(metric_names):
        plt.subplot(2, 2, i + 1)
        values = [metrics_dict[model][metric] for model in model_names]

        # Crear gráfico de barras
        bars = plt.bar(model_names, values)

        # Añadir valores sobre las barras
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.4f}",
                ha="center",
                va="bottom",
            )

        plt.title(metric.capitalize())
        plt.ylabel("Value")
        plt.ylim(0, max(values) * 1.2)  # Ajustar límite vertical

    plt.tight_layout()
    plt.savefig("model_comparison.png")
    plt.close()


def analyze_hyperparameters(
    model_class,
    train_loader,
    val_loader,
    test_loader,
    text_processor,
    param_name,
    param_values,
    fixed_params,
    n_epochs,
    device,
):
    """
    Analiza el impacto de un hiperparámetro específico
    """
    results = {}

    for value in param_values:
        print(f"\nEntrenando modelo con {param_name}={value}")

        # Crear modelo con el valor actual del hiperparámetro
        params = fixed_params.copy()
        params[param_name] = value

        if model_class.__name__ == "TransformerModel":
            model = model_class(
                input_dim=text_processor.vocab_size,
                emb_dim=params["emb_dim"],
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                n_layers=params["n_layers"],
                n_heads=params["n_heads"],
                dropout=params["dropout"],
            ).to(device)
        else:
            model = model_class(
                input_dim=text_processor.vocab_size,
                emb_dim=params["emb_dim"],
                hidden_dim=params["hidden_dim"],
                output_dim=params["output_dim"],
                n_layers=params["n_layers"],
                dropout=params["dropout"],
            ).to(device)

        # Crear optimizador
        optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])

        # Criterio de pérdida
        criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Entrenar modelo
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            n_epochs=n_epochs,
            device=device,
            model_name=f"{model_class.__name__}_{param_name}_{value}",
        )

        # Evaluar modelo
        metrics = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            idx2word=text_processor.idx2word,
        )

        # Guardar resultados
        results[value] = {"metrics": metrics, "history": history}

    # Visualizar resultados
    plt.figure(figsize=(15, 10))

    # Métricas a visualizar
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]

    for i, metric in enumerate(metrics_to_plot):
        plt.subplot(2, 2, i + 1)

        values = [
            results[param_value]["metrics"][metric] for param_value in param_values
        ]

        plt.plot(param_values, values, "o-", linewidth=2)
        plt.title(f"Impact of {param_name} on {metric.capitalize()}")
        plt.xlabel(param_name)
        plt.ylabel(metric.capitalize())
        plt.grid(True)

        # Añadir valores sobre los puntos
        for j, val in enumerate(values):
            plt.text(param_values[j], val + 0.01, f"{val:.4f}", ha="center")

    plt.tight_layout()
    plt.savefig(f"impact_{param_name}.png")
    plt.close()

    return results


def analyze_examples(model, dataloader, text_processor, device, num_examples=5):
    """
    Analiza ejemplos específicos para entender el comportamiento del modelo
    """
    model.eval()
    examples = []

    with torch.no_grad():
        for src, trg in dataloader:
            if len(examples) >= num_examples:
                break

            src, trg = src.to(device), trg.to(device)
            output = model(src)

            # Obtener predicciones
            predictions = torch.argmax(output, dim=2)

            # Analizar cada ejemplo en el batch
            for i in range(src.size(0)):
                if len(examples) >= num_examples:
                    break

                input_text = text_processor.indices_to_text(src[i].cpu().numpy())
                target_text = text_processor.indices_to_text(trg[i].cpu().numpy())
                pred_text = text_processor.indices_to_text(predictions[i].cpu().numpy())

                examples.append(
                    {
                        "input": input_text,
                        "target": target_text,
                        "prediction": pred_text,
                    }
                )

    # Mostrar ejemplos
    print("\nAnálisis de ejemplos específicos:")
    for i, example in enumerate(examples):
        print(f"\nEjemplo {i+1}:")
        print(f"Entrada: {example['input']}")
        print(f"Objetivo: {example['target']}")
        print(f"Predicción: {example['prediction']}")

    return examples


# Configuración principal
INPUT_DIM = text_processor.vocab_size
OUTPUT_DIM = text_processor.vocab_size  # Para generación de secuencia a secuencia
EMB_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
N_HEADS = 8  # Para Transformer
DROPOUT = 0.3
LEARNING_RATE = 0.001
N_EPOCHS = 10

# Mover al dispositivo adecuado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Criterio de pérdida (ignorar padding)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ======= PARTE 1: MODELOS RNN/LSTM =======
print("\n===== PARTE 1: MODELOS RNN/LSTM =====")

# Crear modelos
print("\nCreando modelos RNN/LSTM...")

# Modelo RNN simple
rnn_model = SimpleRNN(
    input_dim=INPUT_DIM,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(device)

# Modelo LSTM
lstm_model = LSTM(
    input_dim=INPUT_DIM,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(device)

# Modelo GRU
gru_model = GRU(
    input_dim=INPUT_DIM,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(device)

# Optimizadores
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
optimizer_gru = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)

# Entrenar modelos
print("\nEntrenando modelo RNN...")
rnn_model, rnn_history = train_model(
    model=rnn_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer_rnn,
    criterion=criterion,
    n_epochs=N_EPOCHS,
    device=device,
    model_name="RNN",
)

print("\nEntrenando modelo LSTM...")
lstm_model, lstm_history = train_model(
    model=lstm_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer_lstm,
    criterion=criterion,
    n_epochs=N_EPOCHS,
    device=device,
    model_name="LSTM",
)

print("\nEntrenando modelo GRU...")
gru_model, gru_history = train_model(
    model=gru_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer_gru,
    criterion=criterion,
    n_epochs=N_EPOCHS,
    device=device,
    model_name="GRU",
)

# Visualizar historiales de entrenamiento
plot_training_history(rnn_history, "RNN")
plot_training_history(lstm_history, "LSTM")
plot_training_history(gru_history, "GRU")

# Evaluar modelos
print("\nEvaluando modelo RNN...")
rnn_metrics = evaluate_model(
    rnn_model, test_loader, criterion, device, text_processor.idx2word
)

print("\nEvaluando modelo LSTM...")
lstm_metrics = evaluate_model(
    lstm_model, test_loader, criterion, device, text_processor.idx2word
)

print("\nEvaluando modelo GRU...")
gru_metrics = evaluate_model(
    gru_model, test_loader, criterion, device, text_processor.idx2word
)

# Comparar modelos RNN/LSTM
rnn_lstm_metrics = {"RNN": rnn_metrics, "LSTM": lstm_metrics, "GRU": gru_metrics}

compare_models(
    metrics_dict=rnn_lstm_metrics,
    model_names=["RNN", "LSTM", "GRU"],
    metric_names=["accuracy", "precision", "recall", "f1"],
)

# Analizar ejemplos específicos
print("\nAnalizando ejemplos específicos con el modelo LSTM...")
lstm_examples = analyze_examples(lstm_model, test_loader, text_processor, device)

# Analizar impacto de hiperparámetros
print("\nAnalizando impacto de hiperparámetros en el modelo LSTM...")

# Parámetros fijos
fixed_params = {
    "emb_dim": EMB_DIM,
    "hidden_dim": HIDDEN_DIM,
    "output_dim": OUTPUT_DIM,
    "n_layers": N_LAYERS,
    "dropout": DROPOUT,
    "learning_rate": LEARNING_RATE,
    "n_heads": N_HEADS,  # Solo para Transformer
}

# Analizar impacto del número de capas
n_layers_values = [1, 2, 3, 4]
n_layers_results = analyze_hyperparameters(
    model_class=LSTM,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    text_processor=text_processor,
    param_name="n_layers",
    param_values=n_layers_values,
    fixed_params=fixed_params,
    n_epochs=5,  # Reducir épocas para agilizar
    device=device,
)

# Analizar impacto de la tasa de aprendizaje
lr_values = [0.0001, 0.001, 0.01, 0.1]
lr_results = analyze_hyperparameters(
    model_class=LSTM,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    text_processor=text_processor,
    param_name="learning_rate",
    param_values=lr_values,
    fixed_params=fixed_params,
    n_epochs=5,  # Reducir épocas para agilizar
    device=device,
)

# ======= PARTE 2: MODELO TRANSFORMER =======
print("\n===== PARTE 2: MODELO TRANSFORMER =======")

# Crear modelo Transformer
print("\nCreando modelo Transformer...")
transformer_model = TransformerModel(
    input_dim=INPUT_DIM,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    dropout=DROPOUT,
).to(device)

# Optimizador
optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)

# Entrenar modelo
print("\nEntrenando modelo Transformer...")
transformer_model, transformer_history = train_model(
    model=transformer_model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer_transformer,
    criterion=criterion,
    n_epochs=N_EPOCHS,
    device=device,
    model_name="Transformer",
)

# Visualizar historial de entrenamiento
plot_training_history(transformer_history, "Transformer")

# Evaluar modelo
print("\nEvaluando modelo Transformer...")
transformer_metrics = evaluate_model(
    transformer_model, test_loader, criterion, device, text_processor.idx2word
)

# Comparar todos los modelos
all_metrics = {
    "RNN": rnn_metrics,
    "LSTM": lstm_metrics,
    "GRU": gru_metrics,
    "Transformer": transformer_metrics,
}

compare_models(
    metrics_dict=all_metrics,
    model_names=["RNN", "LSTM", "GRU", "Transformer"],
    metric_names=["accuracy", "precision", "recall", "f1"],
)

# Comparar BLEU y ROUGE para modelos de generación
plt.figure(figsize=(12, 6))

# Métricas a visualizar
nlp_metrics = ["bleu", "rouge-1", "rouge-2", "rouge-l"]
model_names = ["RNN", "LSTM", "GRU", "Transformer"]

for i, metric in enumerate(nlp_metrics):
    plt.subplot(2, 2, i + 1)
    values = [all_metrics[model][metric] for model in model_names]

    # Crear gráfico de barras
    bars = plt.bar(model_names, values)

    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.title(metric.upper())
    plt.ylabel("Value")
    plt.ylim(0, max(values) * 1.2)  # Ajustar límite vertical

plt.tight_layout()
plt.savefig("nlp_metrics_comparison.png")
plt.close()

# Analizar ejemplos específicos con Transformer
print("\nAnalizando ejemplos específicos con el modelo Transformer...")
transformer_examples = analyze_examples(
    transformer_model, test_loader, text_processor, device
)

# Analizar impacto de hiperparámetros en Transformer
print("\nAnalizando impacto de hiperparámetros en el modelo Transformer...")

# Analizar impacto del número de capas
n_layers_transformer_results = analyze_hyperparameters(
    model_class=TransformerModel,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    text_processor=text_processor,
    param_name="n_layers",
    param_values=n_layers_values,
    fixed_params=fixed_params,
    n_epochs=5,  # Reducir épocas para agilizar
    device=device,
)

# Analizar impacto del número de cabezas de atención
n_heads_values = [2, 4, 8, 16]
n_heads_results = analyze_hyperparameters(
    model_class=TransformerModel,
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    text_processor=text_processor,
    param_name="n_heads",
    param_values=n_heads_values,
    fixed_params=fixed_params,
    n_epochs=5,  # Reducir épocas para agilizar
    device=device,
)

# ======= ANÁLISIS COMPARATIVO FINAL =======
print("\n===== ANÁLISIS COMPARATIVO FINAL =====")


# Comparar tiempos de inferencia
def measure_inference_time(model, dataloader, device, num_batches=10):
    model.eval()
    total_time = 0
    total_samples = 0

    with torch.no_grad():
        for i, (src, _) in enumerate(dataloader):
            if i >= num_batches:
                break

            src = src.to(device)
            batch_size = src.size(0)

            # Medir tiempo
            start_time = time.time()
            _ = model(src)
            end_time = time.time()

            total_time += end_time - start_time
            total_samples += batch_size

    # Tiempo promedio por muestra
    avg_time = total_time / total_samples
    return avg_time


print("\nMidiendo tiempos de inferencia...")
rnn_time = measure_inference_time(rnn_model, test_loader, device)
lstm_time = measure_inference_time(lstm_model, test_loader, device)
gru_time = measure_inference_time(gru_model, test_loader, device)
transformer_time = measure_inference_time(transformer_model, test_loader, device)

# Normalizar tiempos (relativo al más rápido)
min_time = min(rnn_time, lstm_time, gru_time, transformer_time)
relative_times = {
    "RNN": rnn_time / min_time,
    "LSTM": lstm_time / min_time,
    "GRU": gru_time / min_time,
    "Transformer": transformer_time / min_time,
}

print(f"Tiempos de inferencia relativos (menor es mejor):")
for model_name, rel_time in relative_times.items():
    print(f"{model_name}: {rel_time:.2f}x")

# Visualizar tiempos de inferencia
plt.figure(figsize=(10, 6))
plt.bar(relative_times.keys(), relative_times.values())
plt.title("Tiempo de inferencia relativo (menor es mejor)")
plt.ylabel("Tiempo relativo")
plt.grid(True, axis="y")

# Añadir valores sobre las barras
for i, (model, time) in enumerate(relative_times.items()):
    plt.text(i, time + 0.05, f"{time:.2f}x", ha="center")

plt.tight_layout()
plt.savefig("inference_times.png")
plt.close()

# Resumen final de resultados
print("\nResumen final de resultados:")
print("\nMétricas de evaluación:")
for model_name in ["RNN", "LSTM", "GRU", "Transformer"]:
    print(f"\n{model_name}:")
    for metric, value in all_metrics[model_name].items():
        print(f"  {metric}: {value:.4f}")

# Seleccionar el mejor modelo RNN/LSTM basado en F1-score
best_rnn_lstm_model = max(["RNN", "LSTM", "GRU"], key=lambda x: all_metrics[x]["f1"])
print(
    f"\nMejor modelo RNN/LSTM: {best_rnn_lstm_model} (F1: {all_metrics[best_rnn_lstm_model]['f1']:.4f})"
)

# Comparar el mejor modelo RNN/LSTM con Transformer
print("\nComparación del mejor modelo RNN/LSTM vs Transformer:")
print(
    f"F1-score - {best_rnn_lstm_model}: {all_metrics[best_rnn_lstm_model]['f1']:.4f}, Transformer: {all_metrics['Transformer']['f1']:.4f}"
)
print(
    f"BLEU - {best_rnn_lstm_model}: {all_metrics[best_rnn_lstm_model]['bleu']:.4f}, Transformer: {all_metrics['Transformer']['bleu']:.4f}"
)
print(
    f"ROUGE-L - {best_rnn_lstm_model}: {all_metrics[best_rnn_lstm_model]['rouge-l']:.4f}, Transformer: {all_metrics['Transformer']['rouge-l']:.4f}"
)
print(
    f"Tiempo relativo - {best_rnn_lstm_model}: {relative_times[best_rnn_lstm_model]:.2f}x, Transformer: {relative_times['Transformer']:.2f}x"
)

# Visualizar comparación final entre el mejor RNN/LSTM y Transformer
plt.figure(figsize=(15, 10))

# Métricas a visualizar
final_metrics = ["accuracy", "f1", "bleu", "rouge-l"]
final_models = [best_rnn_lstm_model, "Transformer"]

for i, metric in enumerate(final_metrics):
    plt.subplot(2, 2, i + 1)
    values = [all_metrics[model][metric] for model in final_models]

    # Crear gráfico de barras
    bars = plt.bar(final_models, values)

    # Añadir valores sobre las barras
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{height:.4f}",
            ha="center",
            va="bottom",
        )

    plt.title(metric.capitalize())
    plt.ylabel("Value")
    plt.ylim(0, max(values) * 1.2)  # Ajustar límite vertical

plt.tight_layout()
plt.savefig("final_comparison.png")
plt.close()

# Análisis de componentes clave del Transformer
print("\nAnálisis de componentes clave del Transformer:")
print(
    "1. Mecanismo de autoatención: Permite al modelo atender a diferentes partes de la secuencia de entrada simultáneamente."
)
print(
    "2. Codificación posicional: Proporciona información sobre la posición de cada token en la secuencia."
)
print(
    "3. Arquitectura encoder-decoder: Permite procesar la entrada y generar la salida de manera eficiente."
)
print(
    "4. Multi-head attention: Permite al modelo atender a diferentes representaciones del espacio simultáneamente."
)

# Conclusiones
print("\nConclusiones:")
print("1. Comparación de arquitecturas:")
if all_metrics["Transformer"]["f1"] > all_metrics[best_rnn_lstm_model]["f1"]:
    print(
        f"   - El modelo Transformer superó al mejor modelo RNN/LSTM ({best_rnn_lstm_model}) en términos de F1-score."
    )
else:
    print(
        f"   - El mejor modelo RNN/LSTM ({best_rnn_lstm_model}) superó al Transformer en términos de F1-score."
    )

if all_metrics["Transformer"]["bleu"] > all_metrics[best_rnn_lstm_model]["bleu"]:
    print(
        f"   - El modelo Transformer superó al mejor modelo RNN/LSTM en términos de BLEU score."
    )
else:
    print(
        f"   - El mejor modelo RNN/LSTM superó al Transformer en términos de BLEU score."
    )

if relative_times["Transformer"] < relative_times[best_rnn_lstm_model]:
    print(
        f"   - El modelo Transformer fue más rápido en inferencia que el mejor modelo RNN/LSTM."
    )
else:
    print(
        f"   - El mejor modelo RNN/LSTM fue más rápido en inferencia que el Transformer."
    )

print("\n2. Impacto de hiperparámetros:")
print(
    "   - Número de capas: Un mayor número de capas puede mejorar el rendimiento hasta cierto punto, pero también aumenta el riesgo de sobreajuste."
)
print(
    "   - Tasa de aprendizaje: Una tasa de aprendizaje adecuada es crucial para la convergencia del modelo."
)
print(
    "   - Número de cabezas de atención (Transformer): Más cabezas permiten capturar diferentes tipos de relaciones en los datos."
)

print("\n3. Ventajas y desventajas:")
print("   - RNN/LSTM:")
print(
    "     * Ventajas: Más simples, menos parámetros, eficientes para secuencias cortas."
)
print(
    "     * Desventajas: Dificultad para capturar dependencias a largo plazo, procesamiento secuencial."
)
print("   - Transformer:")
print(
    "     * Ventajas: Paralelización, mejor captura de dependencias a largo plazo, atención a diferentes partes de la secuencia."
)
print(
    "     * Desventajas: Mayor número de parámetros, requiere más datos para entrenar efectivamente."
)

print("\nAnálisis completado. Se han generado gráficos para visualizar los resultados.")
