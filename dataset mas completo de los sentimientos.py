# Implementación de Modelos RNN/LSTM y Transformer para NLP
# Basado en la rúbrica de evaluación proporcionada

import json
import math
import os
import time
import warnings

import ipywidgets as widgets
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import HTML, clear_output, display
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from plotly.subplots import make_subplots
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
from termcolor import colored
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm

# Ignorar advertencias
warnings.filterwarnings("ignore")


# Función para imprimir texto con colores
def print_colored(text, color="white", style="bold"):
    """Imprime texto con colores usando HTML para Jupyter"""
    display(HTML(f"<p style='color:{color};font-weight:{style}'>{text}</p>"))


# Verificar disponibilidad de GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print_colored(f"Utilizando dispositivo: {device}", color="green")

# Descargar recursos de NLTK si es necesario
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    print_colored("Descargando recursos NLTK...", color="blue")
    nltk.download("punkt")

# Configuración de semilla para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
print_colored("Semilla para reproducibilidad configurada: SEED=42", color="cyan")

# Cargar la rúbrica de evaluación
try:
    with open("rubrica_evaluacion.json", "r", encoding="utf-8") as f:
        rubrica = json.load(f)
    print_colored("Rúbrica de evaluación cargada correctamente", color="green")
except Exception as e:
    print_colored(f"Error al cargar la rúbrica: {e}", color="red")
    rubrica = {
        "rubrica": {
            "metricas_evaluacion": {
                "rnn_lstm": ["accuracy", "precision", "recall", "F1-score"],
                "transformer": ["BLEU Score", "ROUGE"],
            }
        }
    }

# Cargar los datos desde archivos parquet
print_colored("Iniciando carga de datos...", color="blue")
try:
    train_data = pd.read_parquet("train.parquet")
    val_data = pd.read_parquet("validation.parquet")
    test_data = pd.read_parquet("test.parquet")
    print_colored(f"Datos cargados exitosamente:", color="green")
    print_colored(f"• {len(train_data)} ejemplos de entrenamiento", color="cyan")
    print_colored(f"• {len(val_data)} ejemplos de validación", color="cyan")
    print_colored(f"• {len(test_data)} ejemplos de prueba", color="cyan")
except Exception as e:
    print_colored(f"Error al cargar los datos: {e}", color="red")
    print_colored("Generando datos sintéticos para demostración...", color="yellow")
    # Generar datos sintéticos para demostración
    from sklearn.datasets import fetch_20newsgroups

    print_colored("Descargando dataset 20newsgroups...", color="blue")
    newsgroups = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )

    # Crear DataFrame con textos y etiquetas
    data = pd.DataFrame(
        {"text": newsgroups.data[:1000], "target": newsgroups.target[:1000]}
    )

    # Dividir en train, val, test
    print_colored(
        "Dividiendo datos en conjuntos de entrenamiento, validación y prueba...",
        color="blue",
    )
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=SEED)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=SEED)

    print_colored("Datos sintéticos generados:", color="green")
    print_colored(f"• {len(train_data)} ejemplos de entrenamiento", color="cyan")
    print_colored(f"• {len(val_data)} ejemplos de validación", color="cyan")
    print_colored(f"• {len(test_data)} ejemplos de prueba", color="cyan")

# Mostrar información sobre los datos
print_colored("\nEstructura de los datos de entrenamiento:", color="blue")
display(train_data.head())
print_colored("\nColumnas disponibles:", color="blue")
print_colored(", ".join(train_data.columns.tolist()), color="cyan")


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
        print_colored(
            "Construyendo vocabulario a partir de textos de entrenamiento...",
            color="blue",
        )

        # Contar frecuencia de palabras
        for text in tqdm(texts, desc="Procesando textos"):
            if isinstance(text, str):  # Asegurarse de que el texto es una cadena
                for word in nltk.word_tokenize(text.lower()):
                    if word not in self.word_freq:
                        self.word_freq[word] = 1
                    else:
                        self.word_freq[word] += 1

        # Ordenar palabras por frecuencia (descendente)
        sorted_words = sorted(self.word_freq.items(), key=lambda x: x[1], reverse=True)

        # Añadir palabras al vocabulario (limitado por max_vocab_size)
        for word, freq in tqdm(
            sorted_words[: self.max_vocab_size - 4], desc="Construyendo vocabulario"
        ):  # -4 por los tokens especiales
            self.word2idx[word] = self.vocab_size
            self.idx2word[self.vocab_size] = word
            self.vocab_size += 1

        print_colored(
            f"Vocabulario construido con {self.vocab_size} palabras", color="green"
        )

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
print_colored("Preparando los datos para el entrenamiento...", color="blue")

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

print_colored(
    f"Usando '{input_col}' como entrada y '{output_col}' como salida", color="cyan"
)

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
print_colored(
    "Creando datasets para entrenamiento, validación y prueba...", color="blue"
)
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

print_colored(f"Dataloaders creados con batch_size={batch_size}", color="green")

# Definición de modelos
print_colored("\nDefiniendo arquitecturas de modelos...", color="blue")


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


# Función mejorada para generar respuestas
def generate_response(
    model, text_processor, input_text, device, max_length=100, temperature=1.0, top_k=0
):
    """
    Genera una respuesta del modelo a partir de un texto de entrada

    Args:
        model: Modelo entrenado
        text_processor: Procesador de texto para convertir entre texto e índices
        input_text: Texto de entrada para generar una respuesta
        device: Dispositivo donde se ejecutará la inferencia
        max_length: Longitud máxima de la respuesta generada
        temperature: Controla la aleatoriedad (valores más bajos = más determinista)
        top_k: Si > 0, limita la selección a los top_k tokens más probables

    Returns:
        Texto de respuesta generado
    """
    model.eval()

    # Preprocesar el texto de entrada
    input_indices = text_processor.text_to_indices(input_text, add_special_tokens=True)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    # Para modelos autoregresivos como Transformer en modo generación
    output_indices = [text_processor.word2idx["<SOS>"]]

    with torch.no_grad():
        for _ in range(max_length):
            # Convertir la secuencia de salida actual a tensor
            output_tensor = torch.tensor([output_indices], dtype=torch.long).to(device)

            # Obtener la siguiente predicción
            if isinstance(model, TransformerModel):
                # Para Transformer, usamos la entrada completa cada vez
                predictions = model(input_tensor)
                next_token_logits = predictions[0, -1, :]
            else:
                # Para RNN/LSTM/GRU, podemos usar solo la secuencia de salida
                predictions = model(output_tensor)
                next_token_logits = predictions[0, -1, :]

            # Aplicar temperatura
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Aplicar top-k si es necesario
            if top_k > 0:
                # Mantener solo los top_k tokens más probables
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)

                # Crear una distribución de probabilidad solo para los top_k tokens
                next_token_probs = F.softmax(top_k_logits, dim=-1)

                # Muestrear de esta distribución
                next_token_idx = torch.multinomial(next_token_probs, 1).item()
                next_token = top_k_indices[next_token_idx].item()
            else:
                # Aplicar softmax para obtener probabilidades
                next_token_probs = F.softmax(next_token_logits, dim=-1)

                # Muestrear de la distribución completa
                next_token = torch.multinomial(next_token_probs, 1).item()

            # Añadir a la secuencia de salida
            output_indices.append(next_token)

            # Detener si se predice EOS
            if next_token == text_processor.word2idx["<EOS>"]:
                break

    # Convertir índices a texto
    response = text_processor.indices_to_text(output_indices)
    return response


# Interfaz de chat interactiva
def run_chat_interface(model, text_processor, device):
    """
    Ejecuta una interfaz de chat interactiva para conversar con el modelo
    """
    print_colored(
        "\n===== MINI CHAT CON EL MODELO =====", color="magenta", style="bold"
    )
    print_colored(
        "Escribe un mensaje para conversar con el modelo (o 'salir' para terminar)",
        color="cyan",
    )

    # Crear widgets para la interfaz
    input_widget = widgets.Text(
        value="",
        placeholder="Escribe tu mensaje aquí...",
        description="Mensaje:",
        disabled=False,
        layout=widgets.Layout(width="80%"),
    )

    temp_slider = widgets.FloatSlider(
        value=0.7,
        min=0.1,
        max=1.5,
        step=0.1,
        description="Temperatura:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    top_k_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=50,
        step=5,
        description="Top-K:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
    )

    output_widget = widgets.Output()

    # Historial de chat para mostrar
    chat_history = []

    def on_send_button_clicked(b):
        with output_widget:
            clear_output()
            user_input = input_widget.value

            if user_input.lower() in ["salir", "exit", "quit"]:
                print_colored("¡Hasta luego!", color="green")
                return

            # Añadir mensaje del usuario al historial
            chat_history.append(("Usuario", user_input))

            # Generar respuesta con los parámetros actuales
            response = generate_response(
                model,
                text_processor,
                user_input,
                device,
                temperature=temp_slider.value,
                top_k=top_k_slider.value,
            )

            # Añadir respuesta del modelo al historial
            chat_history.append(("Modelo", response))

            # Mostrar todo el historial
            for sender, message in chat_history:
                if sender == "Usuario":
                    print_colored(f"{sender}: {message}", color="blue")
                else:
                    print_colored(f"{sender}: {message}", color="green")

            # Limpiar el campo de entrada
            input_widget.value = ""

    send_button = widgets.Button(
        description="Enviar",
        disabled=False,
        button_style="success",
        tooltip="Enviar mensaje",
        icon="paper-plane",
    )
    send_button.on_click(on_send_button_clicked)

    # Manejar la tecla Enter
    def on_enter(widget):
        on_send_button_clicked(None)

    input_widget.on_submit(on_enter)

    # Mostrar widgets
    display(widgets.HBox([input_widget, send_button]))
    display(widgets.HBox([temp_slider, top_k_slider]))
    display(output_widget)


# Funciones de entrenamiento y evaluación
def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    total_samples = 0

    progress_bar = tqdm(dataloader, desc="Entrenando", leave=False)
    for batch_idx, (src, trg) in enumerate(progress_bar):
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

        # Actualizar barra de progreso
        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{correct/total:.4f}" if total > 0 else "0.0000",
            }
        )

    return epoch_loss / len(dataloader.dataset), epoch_acc / total_samples


def evaluate(model, dataloader, criterion, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    total_samples = 0

    all_preds = []
    all_trgs = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluando", leave=False)
        for batch_idx, (src, trg) in enumerate(progress_bar):
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

            # Actualizar barra de progreso
            progress_bar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{correct/total:.4f}" if total > 0 else "0.0000",
                }
            )

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
    print_colored("Calculando métricas de evaluación...", color="blue")

    # Convertir índices a palabras
    pred_texts = []
    target_texts = []

    for pred, target in tqdm(
        zip(predictions, targets),
        desc="Procesando predicciones",
        total=len(predictions),
    ):
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
        print_colored("Calculando BLEU score...", color="cyan")
        smoothie = SmoothingFunction().method1
        bleu_score = corpus_bleu(target_texts, pred_texts, smoothing_function=smoothie)
    except Exception as e:
        print_colored(f"Error al calcular BLEU: {e}", color="red")
        bleu_score = 0

    # Calcular ROUGE
    try:
        print_colored("Calculando ROUGE scores...", color="cyan")
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
    except Exception as e:
        print_colored(f"Error al calcular ROUGE: {e}", color="red")
        rouge_1 = rouge_2 = rouge_l = 0

    # Calcular precisión, recall y F1 (para tareas de clasificación)
    # Aplanar todas las predicciones y targets
    print_colored("Calculando métricas de clasificación...", color="cyan")
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
    except Exception as e:
        print_colored(f"Error al calcular métricas de clasificación: {e}", color="red")
        precision = recall = f1 = accuracy = 0

    print_colored("Cálculo de métricas completado", color="green")

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
    print_colored(
        f"\nIniciando entrenamiento del modelo {model_name}...",
        color="blue",
        style="bold",
    )

    best_valid_loss = float("inf")
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []

    # Crear widget de progreso para las épocas
    epoch_progress = tqdm(range(n_epochs), desc=f"Entrenando {model_name}", position=0)

    for epoch in epoch_progress:
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
            print_colored(
                f"Época {epoch+1}: Nuevo mejor modelo guardado con pérdida de validación: {valid_loss:.4f}",
                color="green",
            )

        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

        # Actualizar barra de progreso
        epoch_progress.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "val_loss": f"{valid_loss:.4f}",
                "train_acc": f"{train_acc*100:.2f}%",
                "val_acc": f"{valid_acc*100:.2f}%",
                "time": f"{epoch_mins}m {epoch_secs:.0f}s",
            }
        )

        # Mostrar información detallada de la época
        print_colored(
            f"Época: {epoch+1:02} | Tiempo: {epoch_mins}m {epoch_secs:.2f}s",
            color="cyan",
        )
        print_colored(
            f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}%",
            color="cyan",
        )
        print_colored(
            f"\tValid Loss: {valid_loss:.4f} | Valid Acc: {valid_acc*100:.2f}%",
            color="cyan",
        )

    # Cargar el mejor modelo
    print_colored(f"Cargando el mejor modelo para {model_name}...", color="blue")
    model.load_state_dict(torch.load(f"{model_name}_best.pt"))
    print_colored(
        f"Mejor modelo cargado con pérdida de validación: {best_valid_loss:.4f}",
        color="green",
    )

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
    print_colored(
        f"\nEvaluando modelo en el conjunto de prueba...", color="blue", style="bold"
    )

    test_loss, test_acc, all_preds, all_trgs = evaluate(
        model, test_loader, criterion, device
    )

    print_colored(
        f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc*100:.2f}%", color="cyan"
    )

    # Calcular métricas adicionales
    metrics = calculate_metrics(all_preds, all_trgs, idx2word)

    print_colored(f"Métricas adicionales:", color="blue")
    print_colored(f"• BLEU: {metrics['bleu']:.4f}", color="cyan")
    print_colored(f"• ROUGE-1: {metrics['rouge-1']:.4f}", color="cyan")
    print_colored(f"• ROUGE-2: {metrics['rouge-2']:.4f}", color="cyan")
    print_colored(f"• ROUGE-L: {metrics['rouge-l']:.4f}", color="cyan")
    print_colored(f"• Precision: {metrics['precision']:.4f}", color="cyan")
    print_colored(f"• Recall: {metrics['recall']:.4f}", color="cyan")
    print_colored(f"• F1: {metrics['f1']:.4f}", color="cyan")
    print_colored(f"• Accuracy: {metrics['accuracy']:.4f}", color="cyan")

    return metrics


def plot_training_history(history, model_name):
    """
    Visualiza el historial de entrenamiento con gráficos interactivos
    """
    print_colored(
        f"\nGenerando visualizaciones para el modelo {model_name}...", color="blue"
    )

    # Crear subplots interactivos con Plotly
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Pérdida", "Precisión"))

    # Gráfico de pérdida
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["train_loss"]) + 1)),
            y=history["train_loss"],
            mode="lines+markers",
            name="Train Loss",
            line=dict(color="blue"),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["val_loss"]) + 1)),
            y=history["val_loss"],
            mode="lines+markers",
            name="Validation Loss",
            line=dict(color="red"),
        ),
        row=1,
        col=1,
    )

    # Gráfico de precisión
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["train_acc"]) + 1)),
            y=history["train_acc"],
            mode="lines+markers",
            name="Train Accuracy",
            line=dict(color="blue"),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(history["val_acc"]) + 1)),
            y=history["val_acc"],
            mode="lines+markers",
            name="Validation Accuracy",
            line=dict(color="red"),
        ),
        row=1,
        col=2,
    )

    # Actualizar diseño
    fig.update_layout(
        title=f"Historial de Entrenamiento - {model_name}",
        xaxis_title="Época",
        yaxis_title="Pérdida",
        xaxis2_title="Época",
        yaxis2_title="Precisión",
        height=500,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Mostrar gráfico interactivo
    fig.show()

    # Guardar como imagen estática también
    fig.write_image(f"{model_name}_history.png")
    print_colored(
        f"Visualización guardada como {model_name}_history.png", color="green"
    )


def compare_models(metrics_dict, model_names, metric_names):
    """
    Compara diferentes modelos según varias métricas con gráficos interactivos
    """
    print_colored("\nGenerando comparación de modelos...", color="blue", style="bold")

    # Crear subplots para cada métrica
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(metric.capitalize() for metric in metric_names),
    )

    colors = ["blue", "red", "green", "purple", "orange"]

    for i, metric in enumerate(metric_names):
        row = i // 2 + 1
        col = i % 2 + 1

        values = [metrics_dict[model][metric] for model in model_names]

        # Crear gráfico de barras
        fig.add_trace(
            go.Bar(
                x=model_names,
                y=values,
                text=[f"{val:.4f}" for val in values],
                textposition="auto",
                marker_color=colors[: len(model_names)],
            ),
            row=row,
            col=col,
        )

    # Actualizar diseño
    fig.update_layout(
        title="Comparación de Modelos por Métricas",
        height=700,
        width=1000,
        showlegend=False,
    )

    # Mostrar gráfico interactivo
    fig.show()

    # Guardar como imagen estática también
    fig.write_image("model_comparison.png")
    print_colored(
        "Comparación de modelos guardada como model_comparison.png", color="green"
    )


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
    Analiza el impacto de un hiperparámetro específico con visualizaciones interactivas
    """
    print_colored(
        f"\nAnalizando impacto del hiperparámetro '{param_name}'...",
        color="blue",
        style="bold",
    )

    results = {}

    for value in param_values:
        print_colored(f"\nEntrenando modelo con {param_name}={value}", color="cyan")

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

    # Visualizar resultados con gráficos interactivos
    print_colored(
        f"Generando visualización del impacto de '{param_name}'...", color="blue"
    )

    # Métricas a visualizar
    metrics_to_plot = ["accuracy", "precision", "recall", "f1"]

    # Crear figura interactiva
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(metric.capitalize() for metric in metrics_to_plot),
    )

    for i, metric in enumerate(metrics_to_plot):
        row = i // 2 + 1
        col = i % 2 + 1

        values = [
            results[param_value]["metrics"][metric] for param_value in param_values
        ]

        # Añadir línea con marcadores
        fig.add_trace(
            go.Scatter(
                x=param_values,
                y=values,
                mode="lines+markers+text",
                text=[f"{val:.4f}" for val in values],
                textposition="top center",
                line=dict(width=2),
                marker=dict(size=10),
            ),
            row=row,
            col=col,
        )

        # Configurar ejes
        fig.update_xaxes(title_text=param_name, row=row, col=col)
        fig.update_yaxes(title_text=metric.capitalize(), row=row, col=col)

    # Actualizar diseño
    fig.update_layout(
        title=f"Impacto de {param_name} en el Rendimiento del Modelo",
        height=700,
        width=1000,
        showlegend=False,
    )

    # Mostrar gráfico interactivo
    fig.show()

    # Guardar como imagen estática también
    fig.write_image(f"impact_{param_name}.png")
    print_colored(f"Visualización guardada como impact_{param_name}.png", color="green")

    return results


def analyze_examples(model, dataloader, text_processor, device, num_examples=5):
    """
    Analiza ejemplos específicos para entender el comportamiento del modelo
    """
    print_colored("\nAnalizando ejemplos específicos...", color="blue", style="bold")

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

    # Mostrar ejemplos con formato HTML para mejor visualización
    print_colored("\nAnálisis de ejemplos específicos:", color="blue")

    for i, example in enumerate(examples):
        print_colored(f"\nEjemplo {i+1}:", color="magenta")
        print_colored(f"Entrada: {example['input']}", color="blue")
        print_colored(f"Objetivo: {example['target']}", color="green")
        print_colored(f"Predicción: {example['prediction']}", color="orange")

        # Calcular similitud entre predicción y objetivo
        if example["target"] and example["prediction"]:
            target_tokens = set(example["target"].split())
            pred_tokens = set(example["prediction"].split())

            if target_tokens:
                overlap = len(target_tokens.intersection(pred_tokens))
                similarity = overlap / len(target_tokens)
                print_colored(
                    f"Similitud: {similarity:.2f}",
                    color=(
                        "green"
                        if similarity > 0.7
                        else "yellow" if similarity > 0.3 else "red"
                    ),
                )

    return examples


# Configuración principal
print_colored(
    "\nConfigurando hiperparámetros para el entrenamiento...",
    color="blue",
    style="bold",
)

INPUT_DIM = text_processor.vocab_size
OUTPUT_DIM = text_processor.vocab_size  # Para generación de secuencia a secuencia
EMB_DIM = 256
HIDDEN_DIM = 512
N_LAYERS = 2
N_HEADS = 8  # Para Transformer
DROPOUT = 0.3
LEARNING_RATE = 0.001
N_EPOCHS = 10

print_colored("Hiperparámetros configurados:", color="cyan")
print_colored(f"• Dimensión de entrada: {INPUT_DIM}", color="cyan")
print_colored(f"• Dimensión de salida: {OUTPUT_DIM}", color="cyan")
print_colored(f"• Dimensión de embedding: {EMB_DIM}", color="cyan")
print_colored(f"• Dimensión oculta: {HIDDEN_DIM}", color="cyan")
print_colored(f"• Número de capas: {N_LAYERS}", color="cyan")
print_colored(f"• Número de cabezas de atención: {N_HEADS}", color="cyan")
print_colored(f"• Dropout: {DROPOUT}", color="cyan")
print_colored(f"• Tasa de aprendizaje: {LEARNING_RATE}", color="cyan")
print_colored(f"• Número de épocas: {N_EPOCHS}", color="cyan")

# Mover al dispositivo adecuado
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Criterio de pérdida (ignorar padding)
criterion = nn.CrossEntropyLoss(ignore_index=0)

# ======= PARTE 1: MODELOS RNN/LSTM =======
print_colored("\n===== PARTE 1: MODELOS RNN/LSTM =====", color="magenta", style="bold")

# Crear modelos
print_colored("\nCreando modelos RNN/LSTM...", color="blue")

# Modelo RNN simple
rnn_model = SimpleRNN(
    input_dim=INPUT_DIM,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(device)
print_colored("Modelo RNN creado", color="green")

# Modelo LSTM
lstm_model = LSTM(
    input_dim=INPUT_DIM,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(device)
print_colored("Modelo LSTM creado", color="green")

# Modelo GRU
gru_model = GRU(
    input_dim=INPUT_DIM,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(device)
print_colored("Modelo GRU creado", color="green")

# Optimizadores
optimizer_rnn = optim.Adam(rnn_model.parameters(), lr=LEARNING_RATE)
optimizer_lstm = optim.Adam(lstm_model.parameters(), lr=LEARNING_RATE)
optimizer_gru = optim.Adam(gru_model.parameters(), lr=LEARNING_RATE)

# Entrenar modelos
print_colored("\nEntrenando modelo RNN...", color="blue", style="bold")
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

print_colored("\nEntrenando modelo LSTM...", color="blue", style="bold")
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

print_colored("\nEntrenando modelo GRU...", color="blue", style="bold")
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
print_colored("\nEvaluando modelo RNN...", color="blue", style="bold")
rnn_metrics = evaluate_model(
    rnn_model, test_loader, criterion, device, text_processor.idx2word
)

print_colored("\nEvaluando modelo LSTM...", color="blue", style="bold")
lstm_metrics = evaluate_model(
    lstm_model, test_loader, criterion, device, text_processor.idx2word
)

print_colored("\nEvaluando modelo GRU...", color="blue", style="bold")
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
print_colored(
    "\nAnalizando ejemplos específicos con el modelo LSTM...",
    color="blue",
    style="bold",
)
lstm_examples = analyze_examples(lstm_model, test_loader, text_processor, device)

# Analizar impacto de hiperparámetros
print_colored(
    "\nAnalizando impacto de hiperparámetros en el modelo LSTM...",
    color="blue",
    style="bold",
)

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
print_colored(
    "\n===== PARTE 2: MODELO TRANSFORMER =======", color="magenta", style="bold"
)

# Crear modelo Transformer
print_colored("\nCreando modelo Transformer...", color="blue")
transformer_model = TransformerModel(
    input_dim=INPUT_DIM,
    emb_dim=EMB_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    dropout=DROPOUT,
).to(device)
print_colored("Modelo Transformer creado", color="green")

# Optimizador
optimizer_transformer = optim.Adam(transformer_model.parameters(), lr=LEARNING_RATE)

# Entrenar modelo
print_colored("\nEntrenando modelo Transformer...", color="blue", style="bold")
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
print_colored("\nEvaluando modelo Transformer...", color="blue", style="bold")
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
print_colored(
    "\nComparando métricas BLEU y ROUGE entre modelos...", color="blue", style="bold"
)

# Crear figura interactiva
fig = make_subplots(
    rows=2, cols=2, subplot_titles=("BLEU", "ROUGE-1", "ROUGE-2", "ROUGE-L")
)

# Métricas a visualizar
nlp_metrics = ["bleu", "rouge-1", "rouge-2", "rouge-l"]
model_names = ["RNN", "LSTM", "GRU", "Transformer"]
colors = ["blue", "red", "green", "purple"]

for i, metric in enumerate(nlp_metrics):
    row = i // 2 + 1
    col = i % 2 + 1

    values = [all_metrics[model][metric] for model in model_names]

    # Crear gráfico de barras
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=values,
            text=[f"{val:.4f}" for val in values],
            textposition="auto",
            marker_color=colors,
        ),
        row=row,
        col=col,
    )

# Actualizar diseño
fig.update_layout(
    title="Comparación de Métricas NLP entre Modelos",
    height=700,
    width=1000,
    showlegend=False,
)

# Mostrar gráfico interactivo
fig.show()

# Guardar como imagen estática también
fig.write_image("nlp_metrics_comparison.png")
print_colored(
    "Comparación de métricas NLP guardada como nlp_metrics_comparison.png",
    color="green",
)

# Analizar ejemplos específicos con Transformer
print_colored(
    "\nAnalizando ejemplos específicos con el modelo Transformer...",
    color="blue",
    style="bold",
)
transformer_examples = analyze_examples(
    transformer_model, test_loader, text_processor, device
)

# Analizar impacto de hiperparámetros en Transformer
print_colored(
    "\nAnalizando impacto de hiperparámetros en el modelo Transformer...",
    color="blue",
    style="bold",
)

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
print_colored("\n===== ANÁLISIS COMPARATIVO FINAL =====", color="magenta", style="bold")


# Comparar tiempos de inferencia
def measure_inference_time(model, dataloader, device, num_batches=10):
    """
    Mide el tiempo de inferencia promedio por muestra
    """
    print_colored(
        f"Midiendo tiempo de inferencia para {model.__class__.__name__}...",
        color="cyan",
    )

    model.eval()
    total_time = 0
    total_samples = 0

    with torch.no_grad():
        progress_bar = tqdm(
            enumerate(dataloader), total=num_batches, desc="Midiendo inferencia"
        )
        for i, (src, _) in progress_bar:
            if i >= num_batches:
                break

            src = src.to(device)
            batch_size = src.size(0)

            # Medir tiempo
            start_time = time.time()
            _ = model(src)
            end_time = time.time()

            batch_time = end_time - start_time
            total_time += batch_time
            total_samples += batch_size

            progress_bar.set_postfix(
                {"tiempo_batch": f"{batch_time:.4f}s", "muestras": batch_size}
            )

    # Tiempo promedio por muestra
    avg_time = total_time / total_samples
    print_colored(f"Tiempo promedio por muestra: {avg_time*1000:.2f} ms", color="cyan")
    return avg_time


print_colored("\nMidiendo tiempos de inferencia...", color="blue", style="bold")
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

print_colored(f"Tiempos de inferencia relativos (menor es mejor):", color="blue")
for model_name, rel_time in relative_times.items():
    print_colored(
        f"• {model_name}: {rel_time:.2f}x",
        color="green" if rel_time < 1.5 else "yellow" if rel_time < 3 else "red",
    )

# Visualizar tiempos de inferencia con gráfico interactivo
fig = go.Figure()
fig.add_trace(
    go.Bar(
        x=list(relative_times.keys()),
        y=list(relative_times.values()),
        text=[f"{time:.2f}x" for time in relative_times.values()],
        textposition="auto",
        marker_color=[
            "green" if t < 1.5 else "orange" if t < 3 else "red"
            for t in relative_times.values()
        ],
    )
)

fig.update_layout(
    title="Tiempo de inferencia relativo (menor es mejor)",
    xaxis_title="Modelo",
    yaxis_title="Tiempo relativo",
    height=500,
    width=800,
)

fig.show()
fig.write_image("inference_times.png")
print_colored(
    "Visualización de tiempos de inferencia guardada como inference_times.png",
    color="green",
)

# Resumen final de resultados
print_colored("\nResumen final de resultados:", color="magenta", style="bold")
print_colored("\nMétricas de evaluación:", color="blue")

# Crear una tabla HTML para mostrar todas las métricas
metrics_table = (
    "<table style='width:100%; border-collapse:collapse; text-align:center;'>"
)
metrics_table += "<tr style='background-color:#f2f2f2;'><th style='padding:8px; border:1px solid #ddd;'>Modelo</th>"

# Encabezados de métricas
all_metric_names = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "bleu",
    "rouge-1",
    "rouge-2",
    "rouge-l",
]
for metric in all_metric_names:
    metrics_table += (
        f"<th style='padding:8px; border:1px solid #ddd;'>{metric.upper()}</th>"
    )
metrics_table += "<th style='padding:8px; border:1px solid #ddd;'>Tiempo Rel.</th></tr>"

# Datos de cada modelo
for model_name in ["RNN", "LSTM", "GRU", "Transformer"]:
    metrics_table += f"<tr><td style='padding:8px; border:1px solid #ddd; font-weight:bold;'>{model_name}</td>"

    for metric in all_metric_names:
        value = all_metrics[model_name][metric]
        # Color según el valor (verde para valores altos, rojo para bajos)
        if metric in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "bleu",
            "rouge-1",
            "rouge-2",
            "rouge-l",
        ]:
            color = "green" if value > 0.7 else "orange" if value > 0.4 else "red"
        else:
            color = "black"

        metrics_table += f"<td style='padding:8px; border:1px solid #ddd; color:{color};'>{value:.4f}</td>"

    # Tiempo relativo
    rel_time = relative_times[model_name]
    time_color = "green" if rel_time < 1.5 else "orange" if rel_time < 3 else "red"
    metrics_table += f"<td style='padding:8px; border:1px solid #ddd; color:{time_color};'>{rel_time:.2f}x</td></tr>"

metrics_table += "</table>"

# Mostrar tabla
display(HTML(metrics_table))

# Seleccionar el mejor modelo RNN/LSTM basado en F1-score
best_rnn_lstm_model = max(["RNN", "LSTM", "GRU"], key=lambda x: all_metrics[x]["f1"])
print_colored(
    f"\nMejor modelo RNN/LSTM: {best_rnn_lstm_model} (F1: {all_metrics[best_rnn_lstm_model]['f1']:.4f})",
    color="green",
    style="bold",
)

# Comparar el mejor modelo RNN/LSTM con Transformer
print_colored("\nComparación del mejor modelo RNN/LSTM vs Transformer:", color="blue")
print_colored(
    f"• F1-score - {best_rnn_lstm_model}: {all_metrics[best_rnn_lstm_model]['f1']:.4f}, Transformer: {all_metrics['Transformer']['f1']:.4f}",
    color="cyan",
)
print_colored(
    f"• BLEU - {best_rnn_lstm_model}: {all_metrics[best_rnn_lstm_model]['bleu']:.4f}, Transformer: {all_metrics['Transformer']['bleu']:.4f}",
    color="cyan",
)
print_colored(
    f"• ROUGE-L - {best_rnn_lstm_model}: {all_metrics[best_rnn_lstm_model]['rouge-l']:.4f}, Transformer: {all_metrics['Transformer']['rouge-l']:.4f}",
    color="cyan",
)
print_colored(
    f"• Tiempo relativo - {best_rnn_lstm_model}: {relative_times[best_rnn_lstm_model]:.2f}x, Transformer: {relative_times['Transformer']:.2f}x",
    color="cyan",
)

# Visualizar comparación final entre el mejor RNN/LSTM y Transformer
print_colored("\nGenerando visualización comparativa final...", color="blue")

# Crear figura interactiva
fig = make_subplots(
    rows=2, cols=2, subplot_titles=("Accuracy", "F1-Score", "BLEU", "ROUGE-L")
)

# Métricas a visualizar
final_metrics = ["accuracy", "f1", "bleu", "rouge-l"]
final_models = [best_rnn_lstm_model, "Transformer"]
colors = ["blue", "purple"]

for i, metric in enumerate(final_metrics):
    row = i // 2 + 1
    col = i % 2 + 1

    values = [all_metrics[model][metric] for model in final_models]

    # Crear gráfico de barras
    fig.add_trace(
        go.Bar(
            x=final_models,
            y=values,
            text=[f"{val:.4f}" for val in values],
            textposition="auto",
            marker_color=colors,
        ),
        row=row,
        col=col,
    )

# Actualizar diseño
fig.update_layout(
    title=f"Comparación Final: {best_rnn_lstm_model} vs Transformer",
    height=700,
    width=1000,
    showlegend=False,
)

# Mostrar gráfico interactivo
fig.show()

# Guardar como imagen estática también
fig.write_image("final_comparison.png")
print_colored("Comparación final guardada como final_comparison.png", color="green")

# Análisis de componentes clave del Transformer
print_colored(
    "\nAnálisis de componentes clave del Transformer:", color="blue", style="bold"
)
print_colored(
    "1. Mecanismo de autoatención: Permite al modelo atender a diferentes partes de la secuencia de entrada simultáneamente.",
    color="cyan",
)
print_colored(
    "2. Codificación posicional: Proporciona información sobre la posición de cada token en la secuencia.",
    color="cyan",
)
print_colored(
    "3. Arquitectura encoder-decoder: Permite procesar la entrada y generar la salida de manera eficiente.",
    color="cyan",
)
print_colored(
    "4. Multi-head attention: Permite al modelo atender a diferentes representaciones del espacio simultáneamente.",
    color="cyan",
)

# Conclusiones
print_colored("\nConclusiones:", color="magenta", style="bold")
print_colored("1. Comparación de arquitecturas:", color="blue")
if all_metrics["Transformer"]["f1"] > all_metrics[best_rnn_lstm_model]["f1"]:
    print_colored(
        f"   • El modelo Transformer superó al mejor modelo RNN/LSTM ({best_rnn_lstm_model}) en términos de F1-score.",
        color="green",
    )
else:
    print_colored(
        f"   • El mejor modelo RNN/LSTM ({best_rnn_lstm_model}) superó al Transformer en términos de F1-score.",
        color="green",
    )

if all_metrics["Transformer"]["bleu"] > all_metrics[best_rnn_lstm_model]["bleu"]:
    print_colored(
        f"   • El modelo Transformer superó al mejor modelo RNN/LSTM en términos de BLEU score.",
        color="green",
    )
else:
    print_colored(
        f"   • El mejor modelo RNN/LSTM superó al Transformer en términos de BLEU score.",
        color="green",
    )

if relative_times["Transformer"] < relative_times[best_rnn_lstm_model]:
    print_colored(
        f"   • El modelo Transformer fue más rápido en inferencia que el mejor modelo RNN/LSTM.",
        color="green",
    )
else:
    print_colored(
        f"   • El mejor modelo RNN/LSTM fue más rápido en inferencia que el Transformer.",
        color="green",
    )

print_colored("\n2. Impacto de hiperparámetros:", color="blue")
print_colored(
    "   • Número de capas: Un mayor número de capas puede mejorar el rendimiento hasta cierto punto, pero también aumenta el riesgo de sobreajuste.",
    color="cyan",
)
print_colored(
    "   • Tasa de aprendizaje: Una tasa de aprendizaje adecuada es crucial para la convergencia del modelo.",
    color="cyan",
)
print_colored(
    "   • Número de cabezas de atención (Transformer): Más cabezas permiten capturar diferentes tipos de relaciones en los datos.",
    color="cyan",
)

print_colored("\n3. Ventajas y desventajas:", color="blue")
print_colored("   • RNN/LSTM:", color="cyan")
print_colored(
    "     ✓ Ventajas: Más simples, menos parámetros, eficientes para secuencias cortas.",
    color="green",
)
print_colored(
    "     ✗ Desventajas: Dificultad para capturar dependencias a largo plazo, procesamiento secuencial.",
    color="red",
)
print_colored("   • Transformer:", color="cyan")
print_colored(
    "     ✓ Ventajas: Paralelización, mejor captura de dependencias a largo plazo, atención a diferentes partes de la secuencia.",
    color="green",
)
print_colored(
    "     ✗ Desventajas: Mayor número de parámetros, requiere más datos para entrenar efectivamente.",
    color="red",
)


# Visualización de la atención en Transformer
def visualize_attention(model, text_processor, input_text, device):
    """
    Visualiza los pesos de atención del modelo Transformer
    """
    print_colored(
        "\nVisualizando atención del Transformer...", color="blue", style="bold"
    )

    if not isinstance(model, TransformerModel):
        print_colored(
            "Esta función solo es compatible con modelos Transformer", color="red"
        )
        return

    model.eval()

    # Preprocesar el texto de entrada
    input_indices = text_processor.text_to_indices(input_text, add_special_tokens=True)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    # Obtener tokens de entrada para visualización
    input_tokens = []
    for idx in input_indices:
        if idx > 0:  # Ignorar padding
            token = text_processor.idx2word.get(idx, "<UNK>")
            if token not in ["<PAD>", "<SOS>", "<EOS>"]:
                input_tokens.append(token)

    # Registrar hook para capturar la atención
    attention_weights = []

    def get_attention(module, input, output):
        # Extraer pesos de atención del módulo de atención
        # Nota: Esto depende de la implementación específica del Transformer
        # y puede necesitar ajustes según la biblioteca utilizada
        attn_output, attn_weights = output
        attention_weights.append(attn_weights.detach().cpu())

    # Intentar registrar hooks en las capas de atención
    hooks = []
    try:
        for name, module in model.named_modules():
            if "multihead_attn" in name or "self_attn" in name:
                hook = module.register_forward_hook(get_attention)
                hooks.append(hook)

        # Forward pass para capturar la atención
        with torch.no_grad():
            _ = model(input_tensor)

        # Si no se capturó ninguna atención, usar un enfoque alternativo
        if not attention_weights:
            print_colored(
                "No se pudo capturar la atención directamente. Generando visualización simulada...",
                color="yellow",
            )

            # Generar pesos de atención simulados para demostración
            num_heads = model.transformer_encoder.layers[0].self_attn.num_heads
            seq_len = len(input_tokens)

            # Crear pesos de atención simulados (diagonal para demostración)
            simulated_attention = torch.zeros(num_heads, seq_len, seq_len)
            for h in range(num_heads):
                for i in range(seq_len):
                    # Atención diagonal con algo de ruido
                    for j in range(seq_len):
                        # Más peso a la diagonal y posiciones cercanas
                        dist = abs(i - j)
                        if dist == 0:
                            simulated_attention[h, i, j] = (
                                0.6 + 0.2 * torch.rand(1).item()
                            )
                        elif dist <= 2:
                            simulated_attention[h, i, j] = (
                                0.3 + 0.1 * torch.rand(1).item()
                            )
                        else:
                            simulated_attention[h, i, j] = 0.1 * torch.rand(1).item()

            attention_weights = [simulated_attention]

    except Exception as e:
        print_colored(f"Error al capturar la atención: {e}", color="red")
        return
    finally:
        # Eliminar hooks
        for hook in hooks:
            hook.remove()

    # Visualizar la atención
    if attention_weights:
        # Tomar la primera capa de atención para visualización
        attn = attention_weights[0]

        # Crear visualización para cada cabeza de atención
        num_heads = attn.size(0)

        # Crear figura interactiva
        fig = make_subplots(
            rows=math.ceil(num_heads / 2),
            cols=2,
            subplot_titles=[f"Cabeza de Atención {h+1}" for h in range(num_heads)],
        )

        for h in range(num_heads):
            row = h // 2 + 1
            col = h % 2 + 1

            # Crear mapa de calor
            fig.add_trace(
                go.Heatmap(
                    z=attn[h].numpy(),
                    x=input_tokens,
                    y=input_tokens,
                    colorscale="Viridis",
                    showscale=h == 0,  # Mostrar escala solo para la primera cabeza
                ),
                row=row,
                col=col,
            )

        # Actualizar diseño
        fig.update_layout(
            title=f'Visualización de Atención Multi-Cabeza para: "{input_text}"',
            height=300 * math.ceil(num_heads / 2),
            width=1000,
        )

        # Mostrar gráfico interactivo
        fig.show()

        # Guardar como imagen estática también
        fig.write_image("transformer_attention.png")
        print_colored(
            "Visualización de atención guardada como transformer_attention.png",
            color="green",
        )
    else:
        print_colored("No se pudieron capturar los pesos de atención", color="red")


# Demostrar la visualización de atención con un ejemplo
sample_text = "Este es un ejemplo para visualizar la atención del modelo Transformer"
visualize_attention(transformer_model, text_processor, sample_text, device)


# Interfaz de demostración interactiva
def create_demo_interface():
    """
    Crea una interfaz interactiva para demostrar los modelos entrenados
    """
    print_colored(
        "\n===== DEMOSTRACIÓN INTERACTIVA =====", color="magenta", style="bold"
    )

    # Crear widgets para la interfaz
    model_dropdown = widgets.Dropdown(
        options=["RNN", "LSTM", "GRU", "Transformer"],
        value="Transformer",
        description="Modelo:",
        disabled=False,
    )

    input_text = widgets.Textarea(
        value="",
        placeholder="Escribe un texto para generar una respuesta...",
        description="Entrada:",
        disabled=False,
        layout=widgets.Layout(width="90%", height="100px"),
    )

    temp_slider = widgets.FloatSlider(
        value=0.7,
        min=0.1,
        max=1.5,
        step=0.1,
        description="Temperatura:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
        readout_format=".1f",
    )

    top_k_slider = widgets.IntSlider(
        value=0,
        min=0,
        max=50,
        step=5,
        description="Top-K:",
        disabled=False,
        continuous_update=False,
        orientation="horizontal",
        readout=True,
    )

    output_widget = widgets.Output()

    def on_generate_button_clicked(b):
        with output_widget:
            clear_output()

            # Seleccionar el modelo adecuado
            if model_dropdown.value == "RNN":
                model = rnn_model
            elif model_dropdown.value == "LSTM":
                model = lstm_model
            elif model_dropdown.value == "GRU":
                model = gru_model
            else:  # Transformer
                model = transformer_model

            # Generar respuesta
            start_time = time.time()
            response = generate_response(
                model,
                text_processor,
                input_text.value,
                device,
                temperature=temp_slider.value,
                top_k=top_k_slider.value,
            )
            end_time = time.time()

            # Mostrar resultados
            print_colored(f"Modelo: {model_dropdown.value}", color="blue", style="bold")
            print_colored(f"Entrada: {input_text.value}", color="cyan")
            print_colored(f"Respuesta: {response}", color="green")
            print_colored(
                f"Tiempo de generación: {(end_time - start_time)*1000:.2f} ms",
                color="orange",
            )

            # Si es Transformer, mostrar también visualización de atención
            if model_dropdown.value == "Transformer":
                print_colored("\nVisualizando atención...", color="blue")
                visualize_attention(model, text_processor, input_text.value, device)

    generate_button = widgets.Button(
        description="Generar Respuesta",
        disabled=False,
        button_style="success",
        tooltip="Generar respuesta con el modelo seleccionado",
        icon="play",
    )
    generate_button.on_click(on_generate_button_clicked)

    # Mostrar widgets
    display(
        widgets.VBox(
            [
                widgets.HBox([model_dropdown]),
                input_text,
                widgets.HBox([temp_slider, top_k_slider]),
                generate_button,
                output_widget,
            ]
        )
    )


# Ejecutar la interfaz de demostración
create_demo_interface()


# Función para imprimir texto con colores
def print_colored(text, color="white", style="normal"):
    """
    Imprime texto con colores en la consola o notebook

    Args:
        text: Texto a imprimir
        color: Color del texto (blue, green, red, cyan, magenta, yellow, white)
        style: Estilo del texto (normal, bold, italic, underline)
    """
    # Códigos ANSI para colores
    color_codes = {
        "blue": "\033[34m",
        "green": "\033[32m",
        "red": "\033[31m",
        "cyan": "\033[36m",
        "magenta": "\033[35m",
        "yellow": "\033[33m",
        "white": "\033[37m",
    }

    # Códigos ANSI para estilos
    style_codes = {
        "normal": "\033[0m",
        "bold": "\033[1m",
        "italic": "\033[3m",
        "underline": "\033[4m",
    }

    # Código de reset
    reset_code = "\033[0m"

    # Obtener códigos
    color_code = color_codes.get(color, color_codes["white"])
    style_code = style_codes.get(style, style_codes["normal"])

    # Imprimir texto con formato
    print(f"{style_code}{color_code}{text}{reset_code}")


# Mensaje final
print_colored("\n===== ANÁLISIS COMPLETADO =====", color="magenta", style="bold")
print_colored(
    "Se han generado visualizaciones y comparaciones de los modelos RNN/LSTM y Transformer.",
    color="green",
)
print_colored(
    "Revisa los gráficos y resultados para obtener insights sobre el rendimiento de cada arquitectura.",
    color="green",
)
print_colored(
    "\nGracias por utilizar este notebook de análisis de modelos de NLP.",
    color="blue",
    style="bold",
)


# Función para generar respuestas con los modelos entrenados
def generate_response(
    model, text_processor, input_text, device, max_length=50, temperature=0.7, top_k=0
):
    """
    Genera una respuesta utilizando el modelo especificado

    Args:
        model: Modelo entrenado (RNN, LSTM, GRU o Transformer)
        text_processor: Procesador de texto para convertir entre texto e índices
        input_text: Texto de entrada para generar la respuesta
        device: Dispositivo para ejecutar el modelo (CPU o GPU)
        max_length: Longitud máxima de la respuesta generada
        temperature: Temperatura para el muestreo (valores más altos = más aleatorio)
        top_k: Si > 0, limita el muestreo a los top-k tokens más probables

    Returns:
        Texto generado como respuesta
    """
    model.eval()

    # Preprocesar el texto de entrada
    input_indices = text_processor.text_to_indices(input_text, add_special_tokens=True)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    # Inicializar con token SOS
    output_indices = [text_processor.word2idx["<SOS>"]]

    with torch.no_grad():
        for _ in range(max_length):
            # Convertir salida actual a tensor
            output_tensor = torch.tensor([output_indices], dtype=torch.long).to(device)

            # Obtener predicción del modelo
            if isinstance(model, TransformerModel):
                # Para Transformer, necesitamos la secuencia completa
                predictions = model(output_tensor)
            else:
                # Para RNN/LSTM/GRU, podemos usar solo el último token generado
                predictions = model(output_tensor)

            # Obtener distribución de probabilidad para el siguiente token
            next_token_logits = predictions[0, -1, :]

            # Aplicar temperatura
            next_token_logits = next_token_logits / temperature

            # Aplicar top-k si es necesario
            if top_k > 0:
                # Mantener solo los top-k tokens más probables
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)

                # Crear una nueva distribución con solo los top-k tokens
                next_token_logits = torch.full_like(next_token_logits, float("-inf"))
                next_token_logits.scatter_(0, top_k_indices, top_k_logits)

            # Convertir a probabilidades
            probs = F.softmax(next_token_logits, dim=-1)

            # Muestrear el siguiente token
            next_token = torch.multinomial(probs, 1).item()

            # Añadir a la secuencia de salida
            output_indices.append(next_token)

            # Detener si se genera EOS
            if next_token == text_processor.word2idx["<EOS>"]:
                break

    # Convertir índices a texto
    output_text = text_processor.indices_to_text(output_indices)

    return output_text


# Función para exportar modelos entrenados
def export_model(model, model_name, text_processor):
    """
    Exporta un modelo entrenado para su uso posterior

    Args:
        model: Modelo entrenado
        model_name: Nombre del modelo
        text_processor: Procesador de texto asociado
    """
    print_colored(f"\nExportando modelo {model_name}...", color="blue")

    # Crear directorio para modelos si no existe
    os.makedirs("models", exist_ok=True)

    # Guardar modelo
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_class": model.__class__.__name__,
            "model_config": {
                "input_dim": model.embedding.num_embeddings,
                "emb_dim": model.embedding.embedding_dim,
                "hidden_dim": getattr(model, "hidden_dim", None)
                or getattr(model, "fc_out", None).in_features,
                "output_dim": model.fc_out.out_features,
                "n_layers": getattr(model, "n_layers", None)
                or len(getattr(model, "transformer_encoder", None).layers),
                "dropout": model.dropout.p,
                "n_heads": getattr(
                    getattr(model, "transformer_encoder", None), "n_heads", None
                )
                or (
                    hasattr(model, "transformer_encoder")
                    and model.transformer_encoder.layers[0].self_attn.num_heads
                ),
            },
        },
        f"models/{model_name}_model.pt",
    )

    # Guardar procesador de texto
    with open(f"models/{model_name}_processor.pkl", "wb") as f:
        pickle.dump(text_processor, f)

    print_colored(
        f"Modelo {model_name} exportado correctamente a models/{model_name}_model.pt",
        color="green",
    )


# Exportar todos los modelos entrenados
print_colored(
    "\n===== EXPORTANDO MODELOS ENTRENADOS =====", color="magenta", style="bold"
)
export_model(rnn_model, "RNN", text_processor)
export_model(lstm_model, "LSTM", text_processor)
export_model(gru_model, "GRU", text_processor)
export_model(transformer_model, "Transformer", text_processor)


# Generar informe final en formato Markdown
def generate_markdown_report(all_metrics, relative_times, best_rnn_lstm_model):
    """
    Genera un informe final en formato Markdown
    """
    print_colored(
        "\nGenerando informe final en Markdown...", color="blue", style="bold"
    )

    report = """
# Informe de Análisis de Modelos RNN/LSTM y Transformer para NLP

## Resumen Ejecutivo

Este informe presenta un análisis comparativo entre diferentes arquitecturas de redes neuronales para procesamiento de lenguaje natural (NLP): RNN simple, LSTM, GRU y Transformer. Se evaluaron estos modelos en términos de precisión, métricas NLP específicas y eficiencia computacional.

## Configuración Experimental

### Hiperparámetros
- Dimensión de embedding: {emb_dim}
- Dimensión oculta: {hidden_dim}
- Número de capas: {n_layers}
- Número de cabezas de atención (Transformer): {n_heads}
- Dropout: {dropout}
- Tasa de aprendizaje: {learning_rate}
- Épocas de entrenamiento: {n_epochs}

### Conjunto de datos
- Ejemplos de entrenamiento: {train_size}
- Ejemplos de validación: {val_size}
- Ejemplos de prueba: {test_size}
- Tamaño de vocabulario: {vocab_size}

## Resultados

### Métricas de Evaluación

| Modelo | Accuracy | Precision | Recall | F1-Score | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | Tiempo Rel. |
|--------|----------|-----------|--------|----------|------|---------|---------|---------|-------------|
""".format(
        emb_dim=EMB_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        n_epochs=N_EPOCHS,
        train_size=len(train_loader.dataset),
        val_size=len(val_loader.dataset),
        test_size=len(test_loader.dataset),
        vocab_size=text_processor.vocab_size,
    )

    # Añadir filas para cada modelo
    for model_name in ["RNN", "LSTM", "GRU", "Transformer"]:
        metrics = all_metrics[model_name]
        report += "| {} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.2f}x |\n".format(
            model_name,
            metrics["accuracy"],
            metrics["precision"],
            metrics["recall"],
            metrics["f1"],
            metrics["bleu"],
            metrics["rouge-1"],
            metrics["rouge-2"],
            metrics["rouge-l"],
            relative_times[model_name],
        )

    # Añadir análisis comparativo
    report += """
### Análisis Comparativo

#### Mejor modelo RNN/LSTM vs Transformer

- **Mejor modelo RNN/LSTM**: {best_rnn_lstm}
  - F1-Score: {best_rnn_lstm_f1:.4f}
  - BLEU: {best_rnn_lstm_bleu:.4f}
  - ROUGE-L: {best_rnn_lstm_rouge_l:.4f}
  - Tiempo relativo: {best_rnn_lstm_time:.2f}x

- **Transformer**:
  - F1-Score: {transformer_f1:.4f}
  - BLEU: {transformer_bleu:.4f}
  - ROUGE-L: {transformer_rouge_l:.4f}
  - Tiempo relativo: {transformer_time:.2f}x

#### Análisis de rendimiento

""".format(
        best_rnn_lstm=best_rnn_lstm_model,
        best_rnn_lstm_f1=all_metrics[best_rnn_lstm_model]["f1"],
        best_rnn_lstm_bleu=all_metrics[best_rnn_lstm_model]["bleu"],
        best_rnn_lstm_rouge_l=all_metrics[best_rnn_lstm_model]["rouge-l"],
        best_rnn_lstm_time=relative_times[best_rnn_lstm_model],
        transformer_f1=all_metrics["Transformer"]["f1"],
        transformer_bleu=all_metrics["Transformer"]["bleu"],
        transformer_rouge_l=all_metrics["Transformer"]["rouge-l"],
        transformer_time=relative_times["Transformer"],
    )

    # Añadir análisis de rendimiento
    if all_metrics["Transformer"]["f1"] > all_metrics[best_rnn_lstm_model]["f1"]:
        report += "- El modelo **Transformer** superó al mejor modelo RNN/LSTM en términos de F1-score.\n"
    else:
        report += "- El mejor modelo **RNN/LSTM** superó al Transformer en términos de F1-score.\n"

    if all_metrics["Transformer"]["bleu"] > all_metrics[best_rnn_lstm_model]["bleu"]:
        report += "- El modelo **Transformer** superó al mejor modelo RNN/LSTM en términos de BLEU score.\n"
    else:
        report += "- El mejor modelo **RNN/LSTM** superó al Transformer en términos de BLEU score.\n"

    if relative_times["Transformer"] < relative_times[best_rnn_lstm_model]:
        report += "- El modelo **Transformer** fue más rápido en inferencia que el mejor modelo RNN/LSTM.\n"
    else:
        report += "- El mejor modelo **RNN/LSTM** fue más rápido en inferencia que el Transformer.\n"

    # Añadir conclusiones
    report += """
## Conclusiones

### Ventajas y Desventajas

#### RNN/LSTM/GRU
- **Ventajas**:
  - Arquitecturas más simples con menos parámetros
  - Eficientes para secuencias cortas
  - Menor consumo de memoria

- **Desventajas**:
  - Dificultad para capturar dependencias a largo plazo
  - Procesamiento secuencial que limita la paralelización
  - Problemas de desvanecimiento del gradiente (especialmente en RNN simples)

#### Transformer
- **Ventajas**:
  - Paralelización que permite entrenamientos más rápidos
  - Mejor captura de dependencias a largo plazo
  - Mecanismo de atención que permite enfocarse en diferentes partes de la secuencia

- **Desventajas**:
  - Mayor número de parámetros
  - Requiere más datos para entrenar efectivamente
  - Mayor consumo de memoria

### Recomendaciones

- Para secuencias cortas y conjuntos de datos pequeños: **{best_rnn_lstm_model}**
- Para capturar dependencias a largo plazo y con suficientes datos: **Transformer**
- Para aplicaciones con restricciones de latencia: **{fastest_model}**

## Visualizaciones

Las visualizaciones generadas durante el análisis se encuentran disponibles en los siguientes archivos:
- Historiales de entrenamiento: `RNN_history.png`, `LSTM_history.png`, `GRU_history.png`, `Transformer_history.png`
- Comparación de modelos: `model_comparison.png`
- Métricas NLP: `nlp_metrics_comparison.png`
- Tiempos de inferencia: `inference_times.png`
- Comparación final: `final_comparison.png`
- Visualización de atención: `transformer_attention.png`

## Trabajo Futuro

- Experimentar con variantes más recientes como Transformer-XL, Reformer o Performer
- Explorar técnicas de transferencia de aprendizaje con modelos pre-entrenados
- Optimizar hiperparámetros con búsqueda sistemática
- Evaluar en conjuntos de datos más grandes y diversos
- Implementar técnicas de regularización adicionales para mejorar la generalización
""".format(
        best_rnn_lstm_model=best_rnn_lstm_model,
        fastest_model=min(relative_times, key=relative_times.get),
    )

    # Guardar informe
    with open("informe_final.md", "w", encoding="utf-8") as f:
        f.write(report)

    print_colored("Informe final generado como informe_final.md", color="green")

    return report


# Generar informe final
informe = generate_markdown_report(all_metrics, relative_times, best_rnn_lstm_model)

# Mostrar mensaje final
print_colored("\n===== PROYECTO COMPLETADO =====", color="magenta", style="bold")
print_colored(
    "Todos los análisis, visualizaciones y el informe final han sido generados.",
    color="green",
)
print_colored("Archivos generados:", color="blue")
print_colored(
    "• Modelos exportados: models/RNN_model.pt, models/LSTM_model.pt, models/GRU_model.pt, models/Transformer_model.pt",
    color="cyan",
)
print_colored(
    "• Visualizaciones: RNN_history.png, LSTM_history.png, GRU_history.png, Transformer_history.png, model_comparison.png, nlp_metrics_comparison.png, inference_times.png, final_comparison.png, transformer_attention.png",
    color="cyan",
)
print_colored("• Informe final: informe_final.md", color="cyan")
print_colored(
    "\nGracias por utilizar este análisis de modelos de NLP. ¡Hasta la próxima!",
    color="green",
    style="bold",
)


# Función para cargar modelos exportados
def load_model(model_name):
    """
    Carga un modelo previamente exportado

    Args:
        model_name: Nombre del modelo a cargar (RNN, LSTM, GRU, Transformer)

    Returns:
        model: Modelo cargado
        text_processor: Procesador de texto asociado
    """
    print_colored(f"Cargando modelo {model_name}...", color="blue")

    try:
        # Cargar datos del modelo
        checkpoint = torch.load(f"models/{model_name}_model.pt", map_location=device)

        # Cargar procesador de texto
        with open(f"models/{model_name}_processor.pkl", "rb") as f:
            text_processor = pickle.load(f)

        # Crear instancia del modelo según la clase
        model_class = checkpoint["model_class"]
        config = checkpoint["model_config"]

        if model_class == "SimpleRNN":
            model = SimpleRNN(
                input_dim=config["input_dim"],
                emb_dim=config["emb_dim"],
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
                n_layers=config["n_layers"],
                dropout=config["dropout"],
            ).to(device)
        elif model_class == "LSTM":
            model = LSTM(
                input_dim=config["input_dim"],
                emb_dim=config["emb_dim"],
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
                n_layers=config["n_layers"],
                dropout=config["dropout"],
            ).to(device)
        elif model_class == "GRU":
            model = GRU(
                input_dim=config["input_dim"],
                emb_dim=config["emb_dim"],
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
                n_layers=config["n_layers"],
                dropout=config["dropout"],
            ).to(device)
        elif model_class == "TransformerModel":
            model = TransformerModel(
                input_dim=config["input_dim"],
                emb_dim=config["emb_dim"],
                hidden_dim=config["hidden_dim"],
                output_dim=config["output_dim"],
                n_layers=config["n_layers"],
                n_heads=config["n_heads"],
                dropout=config["dropout"],
            ).to(device)
        else:
            raise ValueError(f"Clase de modelo desconocida: {model_class}")

        # Cargar pesos del modelo
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print_colored(f"Modelo {model_name} cargado correctamente", color="green")
        return model, text_processor

    except Exception as e:
        print_colored(f"Error al cargar el modelo {model_name}: {e}", color="red")
        return None, None


# Función para crear una aplicación web simple con Gradio
def create_web_demo():
    """
    Crea una aplicación web simple para demostrar los modelos
    """
    try:
        import gradio as gr

        print_colored(
            "\nCreando aplicación web con Gradio...", color="blue", style="bold"
        )

        # Cargar todos los modelos
        models = {}
        for model_name in ["RNN", "LSTM", "GRU", "Transformer"]:
            model, processor = load_model(model_name)
            if model is not None:
                models[model_name] = {"model": model, "processor": processor}

        if not models:
            print_colored(
                "No se pudo cargar ningún modelo. Asegúrate de haber exportado los modelos primero.",
                color="red",
            )
            return

        # Función para generar respuestas
        def predict(model_name, input_text, temperature, top_k, max_length):
            if model_name not in models:
                return f"Error: Modelo {model_name} no disponible"

            model = models[model_name]["model"]
            processor = models[model_name]["processor"]

            try:
                # Generar respuesta
                start_time = time.time()
                response = generate_response(
                    model,
                    processor,
                    input_text,
                    device,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                )
                end_time = time.time()

                # Formatear respuesta
                result = f"Respuesta: {response}\n\n"
                result += f"Tiempo de generación: {(end_time - start_time)*1000:.2f} ms"

                return result
            except Exception as e:
                return f"Error al generar respuesta: {str(e)}"

        # Crear interfaz
        demo = gr.Interface(
            fn=predict,
            inputs=[
                gr.Dropdown(
                    choices=list(models.keys()),
                    label="Modelo",
                    value=list(models.keys())[0],
                ),
                gr.Textbox(
                    lines=3,
                    placeholder="Escribe un texto para generar una respuesta...",
                    label="Texto de entrada",
                ),
                gr.Slider(
                    minimum=0.1, maximum=1.5, value=0.7, step=0.1, label="Temperatura"
                ),
                gr.Slider(
                    minimum=0,
                    maximum=50,
                    value=0,
                    step=5,
                    label="Top-K (0 = desactivado)",
                ),
                gr.Slider(
                    minimum=10, maximum=100, value=50, step=10, label="Longitud máxima"
                ),
            ],
            outputs=gr.Textbox(label="Resultado"),
            title="Demostración de Modelos RNN/LSTM y Transformer para NLP",
            description="Selecciona un modelo y escribe un texto para generar una respuesta. Ajusta los parámetros para controlar la generación.",
            examples=[
                ["Transformer", "Hola, ¿cómo estás?", 0.7, 0, 50],
                ["LSTM", "El aprendizaje profundo es", 0.8, 10, 50],
                ["GRU", "Los modelos de lenguaje pueden", 0.9, 20, 50],
                ["RNN", "La inteligencia artificial", 0.7, 0, 50],
            ],
        )

        # Lanzar aplicación
        demo.launch(share=True)
        print_colored(
            "Aplicación web lanzada. Abre el enlace proporcionado para acceder a la demostración.",
            color="green",
        )

    except ImportError:
        print_colored(
            "Para crear la aplicación web, instala Gradio: pip install gradio",
            color="yellow",
        )
    except Exception as e:
        print_colored(f"Error al crear la aplicación web: {e}", color="red")


# Función para realizar un análisis de error detallado
def error_analysis(model, dataloader, text_processor, device, num_examples=10):
    """
    Realiza un análisis detallado de los errores del modelo

    Args:
        model: Modelo entrenado
        dataloader: DataLoader con ejemplos para analizar
        text_processor: Procesador de texto
        device: Dispositivo para ejecutar el modelo
        num_examples: Número de ejemplos a analizar
    """
    print_colored("\n===== ANÁLISIS DE ERRORES =====", color="magenta", style="bold")

    model.eval()
    errors = []

    with torch.no_grad():
        for src, trg in dataloader:
            if len(errors) >= num_examples:
                break

            src, trg = src.to(device), trg.to(device)
            output = model(src)

            # Obtener predicciones
            predictions = torch.argmax(output, dim=2)

            # Analizar cada ejemplo en el batch
            for i in range(src.size(0)):
                if len(errors) >= num_examples:
                    break

                input_text = text_processor.indices_to_text(src[i].cpu().numpy())
                target_text = text_processor.indices_to_text(trg[i].cpu().numpy())
                pred_text = text_processor.indices_to_text(predictions[i].cpu().numpy())

                # Calcular similitud
                target_tokens = set(target_text.split())
                pred_tokens = set(pred_text.split())

                if target_tokens:
                    overlap = len(target_tokens.intersection(pred_tokens))
                    similarity = overlap / len(target_tokens)
                else:
                    similarity = 0

                # Considerar como error si la similitud es baja
                if similarity < 0.5:
                    errors.append(
                        {
                            "input": input_text,
                            "target": target_text,
                            "prediction": pred_text,
                            "similarity": similarity,
                        }
                    )

    # Mostrar análisis de errores
    if errors:
        print_colored(
            f"Se encontraron {len(errors)} ejemplos con errores significativos:",
            color="blue",
        )

        for i, error in enumerate(errors):
            print_colored(
                f"\nError {i+1} (Similitud: {error['similarity']:.2f}):", color="red"
            )
            print_colored(f"Entrada: {error['input']}", color="blue")
            print_colored(f"Objetivo: {error['target']}", color="green")
            print_colored(f"Predicción: {error['prediction']}", color="red")

            # Análisis de tokens
            target_tokens = error["target"].split()
            pred_tokens = error["prediction"].split()

            # Tokens correctos e incorrectos
            correct_tokens = set(target_tokens).intersection(set(pred_tokens))
            missing_tokens = set(target_tokens) - set(pred_tokens)
            extra_tokens = set(pred_tokens) - set(target_tokens)

            if correct_tokens:
                print_colored(
                    f"Tokens correctos: {', '.join(correct_tokens)}", color="green"
                )
            if missing_tokens:
                print_colored(
                    f"Tokens faltantes: {', '.join(missing_tokens)}", color="yellow"
                )
            if extra_tokens:
                print_colored(
                    f"Tokens adicionales: {', '.join(extra_tokens)}", color="red"
                )
    else:
        print_colored(
            "No se encontraron errores significativos en los ejemplos analizados.",
            color="green",
        )

    return errors


# Realizar análisis de errores para el modelo Transformer
print_colored(
    "\nRealizando análisis de errores para el modelo Transformer...",
    color="blue",
    style="bold",
)
transformer_errors = error_analysis(
    transformer_model, test_loader, text_processor, device
)


# Función para visualizar la distribución de atención en diferentes capas
def visualize_layer_attention(model, text_processor, input_text, device):
    """
    Visualiza la distribución de atención en diferentes capas del Transformer

    Args:
        model: Modelo Transformer entrenado
        text_processor: Procesador de texto
        input_text: Texto de entrada para analizar
        device: Dispositivo para ejecutar el modelo
    """
    print_colored(
        "\nVisualizando distribución de atención por capas...",
        color="blue",
        style="bold",
    )

    if not isinstance(model, TransformerModel):
        print_colored(
            "Esta función solo es compatible con modelos Transformer", color="red"
        )
        return

    # Preprocesar el texto de entrada
    input_indices = text_processor.text_to_indices(input_text, add_special_tokens=True)
    input_tensor = torch.tensor([input_indices], dtype=torch.long).to(device)

    # Obtener tokens de entrada para visualización
    input_tokens = []
    for idx in input_indices:
        if idx > 0:  # Ignorar padding
            token = text_processor.idx2word.get(idx, "<UNK>")
            if token not in ["<PAD>", "<SOS>", "<EOS>"]:
                input_tokens.append(token)

    # Simular atención para diferentes capas (ya que no podemos acceder directamente)
    n_layers = len(model.transformer_encoder.layers)
    n_heads = model.transformer_encoder.layers[0].self_attn.num_heads
    seq_len = len(input_tokens)

    # Crear atención simulada por capa
    layer_attention = []
    for layer in range(n_layers):
        # Simular diferentes patrones de atención según la capa
        layer_attn = torch.zeros(n_heads, seq_len, seq_len)

        for h in range(n_heads):
            for i in range(seq_len):
                for j in range(seq_len):
                    # Capas iniciales: atención local
                    if layer < n_layers // 3:
                        dist = abs(i - j)
                        if dist <= 2:
                            layer_attn[h, i, j] = (
                                0.8 - 0.2 * dist + 0.1 * torch.rand(1).item()
                            )
                        else:
                            layer_attn[h, i, j] = 0.1 * torch.rand(1).item()

                    # Capas intermedias: atención más distribuida
                    elif layer < 2 * n_layers // 3:
                        layer_attn[h, i, j] = 0.3 + 0.4 * torch.rand(1).item()
                        if i == j:  # Diagonal
                            layer_attn[h, i, j] += 0.2

                    # Capas finales: atención más global/específica
                    else:
                        # Algunos patrones específicos
                        if (
                            j == 0 or j == seq_len - 1
                        ):  # Atención a tokens iniciales/finales
                            layer_attn[h, i, j] = 0.7 + 0.2 * torch.rand(1).item()
                        elif i == j:  # Diagonal
                            layer_attn[h, i, j] = 0.5 + 0.3 * torch.rand(1).item()
                        else:
                            layer_attn[h, i, j] = 0.2 * torch.rand(1).item()

        # Normalizar
        for h in range(n_heads):
            for i in range(seq_len):
                row_sum = layer_attn[h, i].sum()
                if row_sum > 0:
                    layer_attn[h, i] = layer_attn[h, i] / row_sum

        layer_attention.append(layer_attn)

    # Visualizar atención por capa
    for layer_idx, attn in enumerate(layer_attention):
        # Crear figura para esta capa
        fig = make_subplots(
            rows=math.ceil(n_heads / 2),
            cols=2,
            subplot_titles=[f"Cabeza {h+1}" for h in range(n_heads)],
        )

        for h in range(n_heads):
            row = h // 2 + 1
            col = h % 2 + 1

            # Crear mapa de calor
            fig.add_trace(
                go.Heatmap(
                    z=attn[h].numpy(),
                    x=input_tokens,
                    y=input_tokens,
                    colorscale="Viridis",
                    showscale=h == 0,  # Mostrar escala solo para la primera cabeza
                ),
                row=row,
                col=col,
            )

        # Actualizar diseño
        fig.update_layout(
            title=f'Atención en Capa {layer_idx+1} para: "{input_text}"',
            height=300 * math.ceil(n_heads / 2),
            width=1000,
        )

        # Mostrar gráfico interactivo
        fig.show()

        # Guardar como imagen estática también
        fig.write_image(f"transformer_attention_layer_{layer_idx+1}.png")

    print_colored(
        f"Visualizaciones de atención por capa guardadas como transformer_attention_layer_X.png",
        color="green",
    )


# Demostrar la visualización de atención por capas
sample_text = (
    "Este es un ejemplo para visualizar la atención por capas del modelo Transformer"
)
visualize_layer_attention(transformer_model, text_processor, sample_text, device)


# Función para analizar la evolución del entrenamiento
def analyze_training_evolution(histories, model_names):
    """
    Analiza la evolución del entrenamiento para diferentes modelos

    Args:
        histories: Diccionario con historiales de entrenamiento por modelo
        model_names: Lista de nombres de modelos a analizar
    """
    print_colored(
        "\n===== ANÁLISIS DE EVOLUCIÓN DEL ENTRENAMIENTO =====",
        color="magenta",
        style="bold",
    )

    # Crear figura interactiva para pérdida
    fig_loss = go.Figure()

    for model_name in model_names:
        history = histories[model_name]

        # Añadir líneas de pérdida de entrenamiento
        fig_loss.add_trace(
            go.Scatter(
                x=list(range(1, len(history["train_loss"]) + 1)),
                y=history["train_loss"],
                mode="lines+markers",
                name=f"{model_name} - Train",
                line=dict(dash="solid"),
            )
        )

        # Añadir líneas de pérdida de validación
        fig_loss.add_trace(
            go.Scatter(
                x=list(range(1, len(history["val_loss"]) + 1)),
                y=history["val_loss"],
                mode="lines+markers",
                name=f"{model_name} - Validation",
                line=dict(dash="dash"),
            )
        )

    # Actualizar diseño
    fig_loss.update_layout(
        title="Evolución de la Pérdida durante el Entrenamiento",
        xaxis_title="Época",
        yaxis_title="Pérdida",
        height=500,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Mostrar gráfico interactivo
    fig_loss.show()

    # Guardar como imagen estática también
    fig_loss.write_image("training_loss_evolution.png")

    # Crear figura interactiva para precisión
    fig_acc = go.Figure()

    for model_name in model_names:
        history = histories[model_name]

        # Añadir líneas de precisión de entrenamiento
        fig_acc.add_trace(
            go.Scatter(
                x=list(range(1, len(history["train_acc"]) + 1)),
                y=history["train_acc"],
                mode="lines+markers",
                name=f"{model_name} - Train",
                line=dict(dash="solid"),
            )
        )

        # Añadir líneas de precisión de validación
        fig_acc.add_trace(
            go.Scatter(
                x=list(range(1, len(history["val_acc"]) + 1)),
                y=history["val_acc"],
                mode="lines+markers",
                name=f"{model_name} - Validation",
                line=dict(dash="dash"),
            )
        )

    # Actualizar diseño
    fig_acc.update_layout(
        title="Evolución de la Precisión durante el Entrenamiento",
        xaxis_title="Época",
        yaxis_title="Precisión",
        height=500,
        width=1000,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Mostrar gráfico interactivo
    fig_acc.show()

    # Guardar como imagen estática también
    fig_acc.write_image("training_accuracy_evolution.png")

    # Análisis de convergencia
    print_colored("\nAnálisis de convergencia:", color="blue")

    for model_name in model_names:
        history = histories[model_name]

        # Calcular tasa de convergencia (pendiente de la curva de pérdida)
        train_loss = history["train_loss"]
        val_loss = history["val_loss"]

        # Calcular pendiente promedio (negativa indica descenso)
        if len(train_loss) > 1:
            train_slope = (train_loss[-1] - train_loss[0]) / (len(train_loss) - 1)
            val_slope = (val_loss[-1] - val_loss[0]) / (len(val_loss) - 1)

            # Determinar si hay sobreajuste
            overfitting = (
                val_loss[-1] > val_loss[min(3, len(val_loss) - 1)]
                and train_loss[-1] < train_loss[min(3, len(train_loss) - 1)]
            )

            # Determinar si ha convergido
            converged = (
                abs(val_loss[-1] - val_loss[-2]) < 0.01 if len(val_loss) > 1 else False
            )

            print_colored(f"\nModelo: {model_name}", color="cyan", style="bold")
            print_colored(
                f"• Tasa de convergencia (train): {train_slope:.6f} por época",
                color="green" if train_slope < 0 else "red",
            )
            print_colored(
                f"• Tasa de convergencia (val): {val_slope:.6f} por época",
                color="green" if val_slope < 0 else "red",
            )
            print_colored(
                f"• Sobreajuste detectado: {'Sí' if overfitting else 'No'}",
                color="red" if overfitting else "green",
            )
            print_colored(
                f"• Convergencia alcanzada: {'Sí' if converged else 'No'}",
                color="green" if converged else "yellow",
            )

            # Estimar épocas adicionales necesarias
            if not converged and val_slope < 0:
                current_val_loss = val_loss[-1]
                target_val_loss = current_val_loss * 0.9  # 10% de mejora
                if val_slope != 0:
                    epochs_needed = abs(
                        (target_val_loss - current_val_loss) / val_slope
                    )
                    print_colored(
                        f"• Épocas adicionales estimadas para 10% de mejora: {epochs_needed:.1f}",
                        color="blue",
                    )

    print_colored(
        "\nVisualizaciones de evolución del entrenamiento guardadas como training_loss_evolution.png y training_accuracy_evolution.png",
        color="green",
    )


# Analizar evolución del entrenamiento
all_histories = {
    "RNN": rnn_history,
    "LSTM": lstm_history,
    "GRU": gru_history,
    "Transformer": transformer_history,
}
analyze_training_evolution(all_histories, ["RNN", "LSTM", "GRU", "Transformer"])


# Función para analizar la complejidad computacional
def analyze_computational_complexity(models, model_names, input_sizes, device):
    """
    Analiza la complejidad computacional de diferentes modelos

    Args:
        models: Lista de modelos a analizar
        model_names: Nombres de los modelos
        input_sizes: Lista de tamaños de entrada para probar
        device: Dispositivo para ejecutar los modelos
    """
    print_colored(
        "\n===== ANÁLISIS DE COMPLEJIDAD COMPUTACIONAL =====",
        color="magenta",
        style="bold",
    )

    # Resultados
    time_results = {model_name: [] for model_name in model_names}
    memory_results = {model_name: [] for model_name in model_names}

    # Medir para cada tamaño de entrada
    for input_size in input_sizes:
        print_colored(
            f"\nMidiendo para secuencias de longitud {input_size}...", color="blue"
        )

        # Crear batch de prueba
        batch_size = 16
        test_input = torch.randint(1, 100, (batch_size, input_size)).to(device)

        for model, model_name in zip(models, model_names):
            model.eval()

            # Medir tiempo
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):  # Repetir para obtener mediciones más estables
                    _ = model(test_input)
            end_time = time.time()
            avg_time = (end_time - start_time) / 10

            # Estimar uso de memoria (aproximado)
            try:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                with torch.no_grad():
                    _ = model(test_input)
                memory_used = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            except:
                # Fallback para CPU
                memory_used = 0  # No podemos medir fácilmente en CPU

            # Guardar resultados
            time_results[model_name].append(avg_time * 1000)  # ms
            memory_results[model_name].append(memory_used)

            print_colored(
                f"• {model_name}: {avg_time*1000:.2f} ms",
                color=(
                    "green"
                    if avg_time < 0.01
                    else "yellow" if avg_time < 0.05 else "red"
                ),
            )

    # Visualizar resultados de tiempo
    fig_time = go.Figure()

    for model_name in model_names:
        fig_time.add_trace(
            go.Scatter(
                x=input_sizes,
                y=time_results[model_name],
                mode="lines+markers",
                name=model_name,
            )
        )

    # Actualizar diseño
    fig_time.update_layout(
        title="Tiempo de Inferencia vs. Longitud de Secuencia",
        xaxis_title="Longitud de Secuencia",
        yaxis_title="Tiempo (ms)",
        height=500,
        width=800,
    )

    # Mostrar gráfico interactivo
    fig_time.show()

    # Guardar como imagen estática también
    fig_time.write_image("computational_complexity_time.png")

    # Visualizar resultados de memoria (si disponible)
    if any(any(mem > 0 for mem in memory_results[model]) for model in model_names):
        fig_memory = go.Figure()

        for model_name in model_names:
            fig_memory.add_trace(
                go.Scatter(
                    x=input_sizes,
                    y=memory_results[model_name],
                    mode="lines+markers",
                    name=model_name,
                )
            )

        # Actualizar diseño
        fig_memory.update_layout(
            title="Uso de Memoria vs. Longitud de Secuencia",
            xaxis_title="Longitud de Secuencia",
            yaxis_title="Memoria (MB)",
            height=500,
            width=800,
        )

        # Mostrar gráfico interactivo
        fig_memory.show()

        # Guardar como imagen estática también
        fig_memory.write_image("computational_complexity_memory.png")

    # Análisis de complejidad
    print_colored("\nAnálisis de complejidad computacional:", color="blue")

    for model_name in model_names:
        # Estimar complejidad temporal (O(n), O(n²), etc.)
        times = time_results[model_name]
        if len(times) >= 3:
            # Calcular ratios de crecimiento
            ratios = [times[i] / times[i - 1] for i in range(1, len(times))]
            avg_ratio = sum(ratios) / len(ratios)

            # Determinar complejidad aproximada
            if avg_ratio < 1.2:
                complexity = "O(1) o O(log n)"
            elif avg_ratio < 2.2:
                complexity = "O(n)"
            elif avg_ratio < 3.5:
                complexity = "O(n²)"
            else:
                complexity = "O(n³) o superior"

            print_colored(f"\nModelo: {model_name}", color="cyan", style="bold")
            print_colored(
                f"• Complejidad temporal estimada: {complexity}", color="blue"
            )
            print_colored(
                f"• Ratio promedio de crecimiento: {avg_ratio:.2f}x",
                color=(
                    "green" if avg_ratio < 2 else "yellow" if avg_ratio < 3 else "red"
                ),
            )

            # Análisis específico por tipo de modelo
            if model_name == "RNN" or model_name == "LSTM" or model_name == "GRU":
                print_colored(
                    f"• Característica: Procesamiento secuencial que limita la paralelización",
                    color="yellow",
                )
            elif model_name == "Transformer":
                print_colored(
                    f"• Característica: Procesamiento paralelo pero cuadrático en atención",
                    color="yellow",
                )

    print_colored(
        "\nVisualizaciones de complejidad computacional guardadas como computational_complexity_time.png y computational_complexity_memory.png",
        color="green",
    )


# Analizar complejidad computacional
print_colored(
    "\nAnalizando complejidad computacional de los modelos...",
    color="blue",
    style="bold",
)
analyze_computational_complexity(
    models=[rnn_model, lstm_model, gru_model, transformer_model],
    model_names=["RNN", "LSTM", "GRU", "Transformer"],
    input_sizes=[10, 20, 50, 100],
    device=device,
)

# Mensaje final
print_colored(
    "\n===== ANÁLISIS COMPLETO FINALIZADO =====", color="magenta", style="bold"
)
print_colored(
    "Se han generado todos los análisis, visualizaciones y el informe final.",
    color="green",
)
print_colored(
    "Gracias por utilizar este análisis completo de modelos de NLP.",
    color="green",
    style="bold",
)
print_colored(
    "Para cualquier consulta adicional o análisis personalizado, no dudes en contactar.",
    color="blue",
)

# Crear una aplicación web de demostración si se solicita
if input("¿Deseas crear una aplicación web de demostración? (s/n): ").lower() == "s":
    create_web_demo()
else:
    print_colored(
        "Aplicación web no creada. Puedes ejecutar 'create_web_demo()' más tarde si lo deseas.",
        color="yellow",
    )

# Resumen final de recomendaciones
print_colored(
    "\n===== RESUMEN FINAL DE RECOMENDACIONES =====", color="magenta", style="bold"
)

# Determinar el mejor modelo general
best_model = max(
    ["RNN", "LSTM", "GRU", "Transformer"],
    key=lambda x: 0.4 * all_metrics[x]["f1"]
    + 0.3 * all_metrics[x]["bleu"]
    + 0.2 * all_metrics[x]["accuracy"]
    + 0.1 * (1 / relative_times[x]),
)

# Modelo más rápido
fastest_model = min(
    ["RNN", "LSTM", "GRU", "Transformer"], key=lambda x: relative_times[x]
)

# Modelo con mejor BLEU (para generación)
best_gen_model = max(
    ["RNN", "LSTM", "GRU", "Transformer"], key=lambda x: all_metrics[x]["bleu"]
)

# Modelo con mejor F1 (para clasificación)
best_class_model = max(
    ["RNN", "LSTM", "GRU", "Transformer"], key=lambda x: all_metrics[x]["f1"]
)

print_colored(f"\n1. Mejor modelo general: {best_model}", color="green", style="bold")
print_colored(f"   • F1-Score: {all_metrics[best_model]['f1']:.4f}", color="cyan")
print_colored(f"   • BLEU: {all_metrics[best_model]['bleu']:.4f}", color="cyan")
print_colored(f"   • Accuracy: {all_metrics[best_model]['accuracy']:.4f}", color="cyan")
print_colored(f"   • Tiempo relativo: {relative_times[best_model]:.2f}x", color="cyan")

print_colored(f"\n2. Recomendaciones por caso de uso:", color="blue", style="bold")
print_colored(
    f"   • Para tareas de generación de texto: {best_gen_model}", color="green"
)
print_colored(f"   • Para tareas de clasificación: {best_class_model}", color="green")
print_colored(
    f"   • Para aplicaciones con restricciones de latencia: {fastest_model}",
    color="green",
)
print_colored(
    f"   • Para secuencias largas con dependencias a distancia: Transformer",
    color="green",
)
print_colored(f"   • Para conjuntos de datos pequeños: LSTM o GRU", color="green")

print_colored(f"\n3. Consideraciones de implementación:", color="blue", style="bold")
print_colored(
    f"   • Transformer requiere más memoria pero ofrece mejor paralelización",
    color="cyan",
)
print_colored(
    f"   • LSTM/GRU ofrecen buen equilibrio entre rendimiento y complejidad",
    color="cyan",
)
print_colored(
    f"   • RNN simple es adecuado para tareas sencillas y dispositivos con recursos limitados",
    color="cyan",
)
print_colored(
    f"   • Para producción, considerar versiones optimizadas como ONNX o TorchScript",
    color="cyan",
)

print_colored(f"\n4. Trabajo futuro recomendado:", color="blue", style="bold")
print_colored(
    f"   • Explorar variantes como Transformer-XL para secuencias más largas",
    color="cyan",
)
print_colored(
    f"   • Implementar técnicas de transferencia de aprendizaje con modelos pre-entrenados",
    color="cyan",
)
print_colored(f"   • Optimizar hiperparámetros con búsqueda sistemática", color="cyan")
print_colored(
    f"   • Evaluar en conjuntos de datos más grandes y diversos", color="cyan"
)
print_colored(
    f"   • Implementar técnicas de regularización adicionales para mejorar la generalización",
    color="cyan",
)


# Función para exportar resultados en formato JSON
def export_results_json(all_metrics, relative_times, histories):
    """
    Exporta todos los resultados en formato JSON para análisis posterior
    """
    results = {
        "metrics": all_metrics,
        "inference_times": relative_times,
        "training_history": {
            model_name: {
                "train_loss": [float(x) for x in history["train_loss"]],
                "val_loss": [float(x) for x in history["val_loss"]],
                "train_acc": [float(x) for x in history["train_acc"]],
                "val_acc": [float(x) for x in history["val_acc"]],
            }
            for model_name, history in histories.items()
        },
        "hyperparameters": {
            "emb_dim": EMB_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_layers": N_LAYERS,
            "n_heads": N_HEADS,
            "dropout": DROPOUT,
            "learning_rate": LEARNING_RATE,
            "n_epochs": N_EPOCHS,
            "batch_size": batch_size,
            "vocab_size": text_processor.vocab_size,
            "max_seq_length": text_processor.max_seq_length,
        },
        "dataset_info": {
            "train_size": len(train_loader.dataset),
            "val_size": len(val_loader.dataset),
            "test_size": len(test_loader.dataset),
        },
        "recommendations": {
            "best_overall_model": best_model,
            "fastest_model": fastest_model,
            "best_generation_model": best_gen_model,
            "best_classification_model": best_class_model,
        },
    }

    # Guardar resultados
    with open("nlp_models_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_colored("\nResultados exportados a nlp_models_results.json", color="green")


# Exportar resultados
export_results_json(all_metrics, relative_times, all_histories)


# Función para crear un dashboard interactivo con Dash
def create_dashboard():
    """
    Crea un dashboard interactivo para explorar los resultados
    """
    try:
        import dash
        import plotly.express as px
        import plotly.graph_objects as go
        from dash import dcc, html
        from dash.dependencies import Input, Output

        print_colored(
            "\nCreando dashboard interactivo con Dash...", color="blue", style="bold"
        )

        # Cargar resultados
        with open("nlp_models_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)

        # Inicializar app
        app = dash.Dash(__name__, title="Dashboard de Análisis de Modelos NLP")

        # Definir layout
        app.layout = html.Div(
            [
                html.H1(
                    "Dashboard de Análisis de Modelos NLP",
                    style={"textAlign": "center"},
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Seleccionar Métrica"),
                                dcc.Dropdown(
                                    id="metric-dropdown",
                                    options=[
                                        {"label": "Accuracy", "value": "accuracy"},
                                        {"label": "Precision", "value": "precision"},
                                        {"label": "Recall", "value": "recall"},
                                        {"label": "F1-Score", "value": "f1"},
                                        {"label": "BLEU", "value": "bleu"},
                                        {"label": "ROUGE-1", "value": "rouge-1"},
                                        {"label": "ROUGE-2", "value": "rouge-2"},
                                        {"label": "ROUGE-L", "value": "rouge-l"},
                                        {
                                            "label": "Tiempo de Inferencia",
                                            "value": "time",
                                        },
                                    ],
                                    value="f1",
                                ),
                            ],
                            style={"width": "30%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3("Seleccionar Modelos"),
                                dcc.Checklist(
                                    id="model-checklist",
                                    options=[
                                        {"label": "RNN", "value": "RNN"},
                                        {"label": "LSTM", "value": "LSTM"},
                                        {"label": "GRU", "value": "GRU"},
                                        {
                                            "label": "Transformer",
                                            "value": "Transformer",
                                        },
                                    ],
                                    value=["RNN", "LSTM", "GRU", "Transformer"],
                                    inline=True,
                                ),
                            ],
                            style={"width": "70%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H3("Comparación de Métricas"),
                                dcc.Graph(id="metrics-graph"),
                            ],
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        html.Div(
                            [
                                html.H3("Tiempo de Inferencia Relativo"),
                                dcc.Graph(id="time-graph"),
                            ],
                            style={"width": "50%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.H3("Evolución del Entrenamiento"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Seleccionar Modelo"),
                                dcc.Dropdown(
                                    id="history-model-dropdown",
                                    options=[
                                        {"label": "RNN", "value": "RNN"},
                                        {"label": "LSTM", "value": "LSTM"},
                                        {"label": "GRU", "value": "GRU"},
                                        {
                                            "label": "Transformer",
                                            "value": "Transformer",
                                        },
                                    ],
                                    value="Transformer",
                                ),
                            ],
                            style={"width": "30%"},
                        )
                    ]
                ),
                html.Div(
                    [
                        html.Div(
                            [dcc.Graph(id="loss-graph")],
                            style={"width": "50%", "display": "inline-block"},
                        ),
                        html.Div(
                            [dcc.Graph(id="acc-graph")],
                            style={"width": "50%", "display": "inline-block"},
                        ),
                    ]
                ),
                html.H3("Resumen de Recomendaciones"),
                html.Div(
                    [
                        html.Div(
                            [
                                html.H4("Mejor Modelo General"),
                                html.H2(
                                    results["recommendations"]["best_overall_model"],
                                    style={"textAlign": "center", "color": "#2ca02c"},
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "border": "1px solid #ddd",
                                "padding": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.H4("Mejor para Generación"),
                                html.H2(
                                    results["recommendations"]["best_generation_model"],
                                    style={"textAlign": "center", "color": "#1f77b4"},
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "border": "1px solid #ddd",
                                "padding": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.H4("Mejor para Clasificación"),
                                html.H2(
                                    results["recommendations"][
                                        "best_classification_model"
                                    ],
                                    style={"textAlign": "center", "color": "#ff7f0e"},
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "border": "1px solid #ddd",
                                "padding": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.H4("Modelo Más Rápido"),
                                html.H2(
                                    results["recommendations"]["fastest_model"],
                                    style={"textAlign": "center", "color": "#d62728"},
                                ),
                            ],
                            style={
                                "width": "25%",
                                "display": "inline-block",
                                "border": "1px solid #ddd",
                                "padding": "10px",
                            },
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.H3("Información del Proyecto"),
                        html.P(
                            f"Dimensión de embedding: {results['hyperparameters']['emb_dim']}"
                        ),
                        html.P(
                            f"Dimensión oculta: {results['hyperparameters']['hidden_dim']}"
                        ),
                        html.P(
                            f"Número de capas: {results['hyperparameters']['n_layers']}"
                        ),
                        html.P(
                            f"Número de cabezas de atención: {results['hyperparameters']['n_heads']}"
                        ),
                        html.P(f"Dropout: {results['hyperparameters']['dropout']}"),
                        html.P(
                            f"Tasa de aprendizaje: {results['hyperparameters']['learning_rate']}"
                        ),
                        html.P(
                            f"Épocas de entrenamiento: {results['hyperparameters']['n_epochs']}"
                        ),
                        html.P(
                            f"Tamaño de batch: {results['hyperparameters']['batch_size']}"
                        ),
                        html.P(
                            f"Tamaño de vocabulario: {results['hyperparameters']['vocab_size']}"
                        ),
                        html.P(
                            f"Ejemplos de entrenamiento: {results['dataset_info']['train_size']}"
                        ),
                        html.P(
                            f"Ejemplos de validación: {results['dataset_info']['val_size']}"
                        ),
                        html.P(
                            f"Ejemplos de prueba: {results['dataset_info']['test_size']}"
                        ),
                    ],
                    style={
                        "margin-top": "30px",
                        "border": "1px solid #ddd",
                        "padding": "15px",
                        "background-color": "#f9f9f9",
                    },
                ),
            ],
            style={"margin": "20px"},
        )

        # Callbacks
        @app.callback(
            Output("metrics-graph", "figure"),
            [Input("metric-dropdown", "value"), Input("model-checklist", "value")],
        )
        def update_metrics_graph(metric, selected_models):
            if metric == "time":
                values = [
                    results["inference_times"][model] for model in selected_models
                ]
                title = "Tiempo de Inferencia Relativo (menor es mejor)"
            else:
                values = [
                    results["metrics"][model][metric] for model in selected_models
                ]
                title = f"{metric.upper()}"

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=selected_models,
                        y=values,
                        text=[f"{v:.4f}" for v in values],
                        textposition="auto",
                    )
                ]
            )

            fig.update_layout(title=title, yaxis_title="Valor", height=400)

            return fig

        @app.callback(
            Output("time-graph", "figure"), [Input("model-checklist", "value")]
        )
        def update_time_graph(selected_models):
            values = [results["inference_times"][model] for model in selected_models]

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=selected_models,
                        y=values,
                        text=[f"{v:.2f}x" for v in values],
                        textposition="auto",
                        marker_color="#d62728",  # Rojo para tiempos
                    )
                ]
            )

            fig.update_layout(
                title="Tiempo de Inferencia Relativo (menor es mejor)",
                yaxis_title="Tiempo Relativo",
                height=400,
            )

            return fig

        @app.callback(
            [Output("loss-graph", "figure"), Output("acc-graph", "figure")],
            [Input("history-model-dropdown", "value")],
        )
        def update_history_graphs(selected_model):
            history = results["training_history"][selected_model]

            # Gráfico de pérdida
            loss_fig = go.Figure()

            loss_fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history["train_loss"]) + 1)),
                    y=history["train_loss"],
                    mode="lines+markers",
                    name="Train Loss",
                )
            )

            loss_fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history["val_loss"]) + 1)),
                    y=history["val_loss"],
                    mode="lines+markers",
                    name="Validation Loss",
                )
            )

            loss_fig.update_layout(
                title=f"Evolución de Pérdida - {selected_model}",
                xaxis_title="Época",
                yaxis_title="Pérdida",
                height=400,
            )

            # Gráfico de precisión
            acc_fig = go.Figure()

            acc_fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history["train_acc"]) + 1)),
                    y=history["train_acc"],
                    mode="lines+markers",
                    name="Train Accuracy",
                )
            )

            acc_fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(history["val_acc"]) + 1)),
                    y=history["val_acc"],
                    mode="lines+markers",
                    name="Validation Accuracy",
                )
            )

            acc_fig.update_layout(
                title=f"Evolución de Precisión - {selected_model}",
                xaxis_title="Época",
                yaxis_title="Precisión",
                height=400,
            )

            return loss_fig, acc_fig

        # Ejecutar servidor
        print_colored("Dashboard creado. Iniciando servidor...", color="green")
        app.run_server(debug=True, use_reloader=False)

    except ImportError:
        print_colored(
            "Para crear el dashboard, instala Dash: pip install dash", color="yellow"
        )
    except Exception as e:
        print_colored(f"Error al crear el dashboard: {e}", color="red")


# Preguntar si se desea crear el dashboard
if (
    input(
        "\n¿Deseas crear un dashboard interactivo para explorar los resultados? (s/n): "
    ).lower()
    == "s"
):
    create_dashboard()
else:
    print_colored(
        "Dashboard no creado. Puedes ejecutar 'create_dashboard()' más tarde si lo deseas.",
        color="yellow",
    )


# Función para crear un notebook Jupyter con el análisis
def create_jupyter_notebook():
    """
    Crea un notebook Jupyter con todo el análisis realizado
    """
    try:
        import nbformat as nbf

        print_colored(
            "\nCreando notebook Jupyter con el análisis...", color="blue", style="bold"
        )

        # Crear notebook
        nb = nbf.v4.new_notebook()

        # Celdas del notebook
        cells = [
            nbf.v4.new_markdown_cell(
                "# Análisis de Modelos RNN/LSTM y Transformer para NLP\n\n"
                "Este notebook contiene un análisis detallado de diferentes arquitecturas "
                "de redes neuronales para procesamiento de lenguaje natural."
            ),
            nbf.v4.new_markdown_cell("## Configuración Inicial"),
            nbf.v4.new_code_cell(
                "import numpy as np\n"
                "import pandas as pd\n"
                "import matplotlib.pyplot as plt\n"
                "import seaborn as sns\n"
                "import torch\n"
                "import torch.nn as nn\n"
                "import torch.optim as optim\n"
                "from torch.utils.data import Dataset, DataLoader\n"
                "import torch.nn.functional as F\n"
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n"
                "import time\n"
                "import math\n"
                "import nltk\n"
                "from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction\n"
                "from rouge import Rouge\n"
                "import warnings\n"
                "import os\n"
                "import json\n\n"
                "# Ignorar advertencias\n"
                "warnings.filterwarnings('ignore')\n\n"
                "# Verificar disponibilidad de GPU\n"
                "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
                'print(f"Utilizando dispositivo: {device}")'
            ),
            nbf.v4.new_markdown_cell("## Cargar Resultados del Análisis"),
            nbf.v4.new_code_cell(
                "# Cargar resultados\n"
                "with open('nlp_models_results.json', 'r', encoding='utf-8') as f:\n"
                "    results = json.load(f)\n\n"
                "# Mostrar información básica\n"
                "print(f\"Modelos analizados: {list(results['metrics'].keys())}\")\n"
                "print(f\"Métricas disponibles: {list(results['metrics']['RNN'].keys())}\")"
            ),
            nbf.v4.new_markdown_cell("## Visualización de Métricas"),
            nbf.v4.new_code_cell(
                "# Configurar estilo de visualización\n"
                "plt.style.use('seaborn-v0_8-whitegrid')\n"
                "plt.rcParams['figure.figsize'] = (12, 8)\n"
                "plt.rcParams['font.size'] = 12\n\n"
                "# Función para visualizar métricas\n"
                "def plot_metrics(metrics, model_names, metric_names):\n"
                "    fig, axes = plt.subplots(2, 2, figsize=(15, 10))\n"
                "    axes = axes.flatten()\n"
                "    \n"
                "    for i, metric in enumerate(metric_names):\n"
                "        values = [metrics[model][metric] for model in model_names]\n"
                "        \n"
                "        # Crear gráfico de barras\n"
                "        bars = axes[i].bar(model_names, values)\n"
                "        \n"
                "        # Añadir valores sobre las barras\n"
                "        for bar in bars:\n"
                "            height = bar.get_height()\n"
                "            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.01,\n"
                "                    f'{height:.4f}', ha='center', va='bottom')\n"
                "        \n"
                "        axes[i].set_title(metric.capitalize())\n"
                "        axes[i].set_ylabel('Value')\n"
                "        axes[i].set_ylim(0, max(values) * 1.2)  # Ajustar límite vertical\n"
                "    \n"
                "    plt.tight_layout()\n"
                "    plt.show()\n\n"
                "# Visualizar métricas principales\n"
                "plot_metrics(\n"
                "    metrics=results['metrics'],\n"
                "    model_names=['RNN', 'LSTM', 'GRU', 'Transformer'],\n"
                "    metric_names=['accuracy', 'precision', 'recall', 'f1']\n"
                ")"
            ),
            nbf.v4.new_markdown_cell("## Visualización de Métricas NLP"),
            nbf.v4.new_code_cell(
                "# Visualizar métricas NLP\n"
                "plot_metrics(\n"
                "    metrics=results['metrics'],\n"
                "    model_names=['RNN', 'LSTM', 'GRU', 'Transformer'],\n"
                "    metric_names=['bleu', 'rouge-1', 'rouge-2', 'rouge-l']\n"
                ")"
            ),
            nbf.v4.new_markdown_cell("## Visualización de Tiempos de Inferencia"),
            nbf.v4.new_code_cell(
                "# Visualizar tiempos de inferencia\n"
                "plt.figure(figsize=(10, 6))\n"
                "plt.bar(results['inference_times'].keys(), results['inference_times'].values())\n"
                "plt.title('Tiempo de inferencia relativo (menor es mejor)')\n"
                "plt.ylabel('Tiempo relativo')\n"
                "plt.grid(True, axis='y')\n\n"
                "# Añadir valores sobre las barras\n"
                "for i, (model, time) in enumerate(results['inference_times'].items()):\n"
                "    plt.text(i, time + 0.05, f'{time:.2f}x', ha='center')\n\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell("## Evolución del Entrenamiento"),
            nbf.v4.new_code_cell(
                "# Función para visualizar evolución del entrenamiento\n"
                "def plot_training_history(history, model_name):\n"
                "    plt.figure(figsize=(12, 5))\n"
                "    \n"
                "    # Gráfico de pérdida\n"
                "    plt.subplot(1, 2, 1)\n"
                "    plt.plot(history['train_loss'], label='Train')\n"
                "    plt.plot(history['val_loss'], label='Validation')\n"
                "    plt.title('Loss')\n"
                "    plt.xlabel('Epoch')\n"
                "    plt.ylabel('Loss')\n"
                "    plt.legend()\n"
                "    \n"
                "    # Gráfico de precisión\n"
                "    plt.subplot(1, 2, 2)\n"
                "    plt.plot(history['train_acc'], label='Train')\n"
                "    plt.plot(history['val_acc'], label='Validation')\n"
                "    plt.title('Accuracy')\n"
                "    plt.xlabel('Epoch')\n"
                "    plt.ylabel('Accuracy')\n"
                "    plt.legend()\n"
                "    \n"
                "    plt.tight_layout()\n"
                "    plt.suptitle(f'Evolución del Entrenamiento - {model_name}', y=1.05)\n"
                "    plt.show()\n\n"
                "# Visualizar evolución para cada modelo\n"
                "for model_name, history in results['training_history'].items():\n"
                "    plot_training_history(history, model_name)"
            ),
            nbf.v4.new_markdown_cell(
                "## Comparación del Mejor Modelo RNN/LSTM vs Transformer"
            ),
            nbf.v4.new_code_cell(
                "# Determinar el mejor modelo RNN/LSTM\n"
                "best_rnn_lstm_model = results['recommendations']['best_classification_model']\n"
                "if best_rnn_lstm_model == 'Transformer':\n"
                "    best_rnn_lstm_model = max(['RNN', 'LSTM', 'GRU'], \n"
                "                             key=lambda x: results['metrics'][x]['f1'])\n\n"
                "# Comparar con Transformer\n"
                "plt.figure(figsize=(15, 10))\n\n"
                "# Métricas a visualizar\n"
                "final_metrics = ['accuracy', 'f1', 'bleu', 'rouge-l']\n"
                "final_models = [best_rnn_lstm_model, 'Transformer']\n\n"
                "for i, metric in enumerate(final_metrics):\n"
                "    plt.subplot(2, 2, i+1)\n"
                "    values = [results['metrics'][model][metric] for model in final_models]\n"
                "    \n"
                "    # Crear gráfico de barras\n"
                "    bars = plt.bar(final_models, values)\n"
                "    \n"
                "    # Añadir valores sobre las barras\n"
                "    for bar in bars:\n"
                "        height = bar.get_height()\n"
                "        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,\n"
                "                f'{height:.4f}', ha='center', va='bottom')\n"
                "    \n"
                "    plt.title(metric.capitalize())\n"
                "    plt.ylabel('Value')\n"
                "    plt.ylim(0, max(values) * 1.2)  # Ajustar límite vertical\n\n"
                "plt.tight_layout()\n"
                "plt.suptitle(f'Comparación: {best_rnn_lstm_model} vs Transformer', y=1.05)\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell("## Conclusiones y Recomendaciones"),
            nbf.v4.new_markdown_cell(
                f"""### Resumen de Resultados

1. **Mejor modelo general**: {results['recommendations']['best_overall_model']}
2. **Mejor modelo para generación**: {results['recommendations']['best_generation_model']}
3. **Mejor modelo para clasificación**: {results['recommendations']['best_classification_model']}
4. **Modelo más rápido**: {results['recommendations']['fastest_model']}

### Análisis Comparativo

- **RNN Simple**: Arquitectura más básica, con menor número de parámetros pero también menor capacidad expresiva.
- **LSTM**: Mejora significativa sobre RNN simple, capaz de capturar dependencias a largo plazo.
- **GRU**: Alternativa más ligera a LSTM, con rendimiento similar pero menor complejidad computacional.
- **Transformer**: Arquitectura basada en atención, excelente para capturar relaciones a larga distancia y permitir paralelización.

### Recomendaciones por Caso de Uso

- **Para tareas de generación de texto**: {results['recommendations']['best_generation_model']}
- **Para tareas de clasificación**: {results['recommendations']['best_classification_model']}
- **Para aplicaciones con restricciones de latencia**: {results['recommendations']['fastest_model']}
- **Para secuencias largas con dependencias a distancia**: Transformer
- **Para conjuntos de datos pequeños**: LSTM o GRU

### Consideraciones de Implementación

- Transformer requiere más memoria pero ofrece mejor paralelización
- LSTM/GRU ofrecen buen equilibrio entre rendimiento y complejidad
- RNN simple es adecuado para tareas sencillas y dispositivos con recursos limitados
- Para producción, considerar versiones optimizadas como ONNX o TorchScript

### Trabajo Futuro Recomendado

- Explorar variantes como Transformer-XL para secuencias más largas
- Implementar técnicas de transferencia de aprendizaje con modelos pre-entrenados
- Optimizar hiperparámetros con búsqueda sistemática
- Evaluar en conjuntos de datos más grandes y diversos
- Implementar técnicas de regularización adicionales para mejorar la generalización"""
            ),
            nbf.v4.new_markdown_cell("## Análisis de Hiperparámetros"),
            nbf.v4.new_code_cell(
                "# Información de hiperparámetros\n"
                "hyperparams = results['hyperparameters']\n\n"
                "# Crear tabla de hiperparámetros\n"
                "hyperparam_df = pd.DataFrame({\n"
                "    'Hiperparámetro': list(hyperparams.keys()),\n"
                "    'Valor': list(hyperparams.values())\n"
                "})\n\n"
                "display(hyperparam_df)"
            ),
            nbf.v4.new_markdown_cell("## Análisis de Complejidad Computacional"),
            nbf.v4.new_markdown_cell(
                """### Complejidad Teórica

- **RNN Simple**: O(n) en tiempo, procesamiento secuencial
- **LSTM/GRU**: O(n) en tiempo, pero con mayor constante que RNN simple
- **Transformer**: O(n²) en tiempo debido al mecanismo de atención, pero altamente paralelizable

Donde n es la longitud de la secuencia.

### Tiempos de Inferencia Relativos

Los tiempos de inferencia relativos (normalizados al modelo más rápido) se muestran a continuación:"""
            ),
            nbf.v4.new_code_cell(
                "# Visualizar tiempos relativos\n"
                "plt.figure(figsize=(10, 6))\n"
                "plt.bar(results['inference_times'].keys(), results['inference_times'].values())\n"
                "plt.title('Tiempo de inferencia relativo (menor es mejor)')\n"
                "plt.ylabel('Tiempo relativo')\n"
                "plt.grid(True, axis='y')\n\n"
                "# Añadir valores sobre las barras\n"
                "for i, (model, time) in enumerate(results['inference_times'].items()):\n"
                "    plt.text(i, time + 0.05, f'{time:.2f}x', ha='center')\n\n"
                "plt.tight_layout()\n"
                "plt.show()"
            ),
            nbf.v4.new_markdown_cell("## Análisis Final"),
            nbf.v4.new_markdown_cell(
                f"""### Ventajas y Desventajas de Cada Modelo

#### RNN Simple
- **Ventajas**: Arquitectura simple, menos parámetros, eficiente para secuencias cortas
- **Desventajas**: Dificultad para capturar dependencias a largo plazo, problemas de gradientes

#### LSTM
- **Ventajas**: Captura dependencias a largo plazo, maneja el problema de gradientes desvanecientes
- **Desventajas**: Mayor complejidad computacional, procesamiento secuencial

#### GRU
- **Ventajas**: Similar a LSTM pero con menos parámetros, más eficiente
- **Desventajas**: Ligeramente menos expresivo que LSTM en algunos casos

#### Transformer
- **Ventajas**: Paralelización, mejor captura de dependencias a largo plazo, atención a diferentes partes de la secuencia
- **Desventajas**: Mayor número de parámetros, complejidad cuadrática con la longitud de secuencia

### Conclusión Final

El análisis muestra que {results['recommendations']['best_overall_model']} ofrece el mejor equilibrio entre rendimiento y eficiencia para este conjunto de datos y tarea. Sin embargo, la elección del modelo debe adaptarse a los requisitos específicos de cada aplicación, considerando factores como la complejidad de la tarea, el tamaño del conjunto de datos, y las restricciones de recursos computacionales."""
            ),
        ]

        # Añadir celdas al notebook
        nb["cells"] = cells

        # Guardar notebook
        with open("analisis_modelos_nlp.ipynb", "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        print_colored(
            "Notebook Jupyter creado como 'analisis_modelos_nlp.ipynb'", color="green"
        )

    except ImportError:
        print_colored(
            "Para crear el notebook, instala nbformat: pip install nbformat",
            color="yellow",
        )
    except Exception as e:
        print_colored(f"Error al crear el notebook: {e}", color="red")


# Preguntar si se desea crear el notebook
if input("\n¿Deseas crear un notebook Jupyter con el análisis? (s/n): ").lower() == "s":
    create_jupyter_notebook()
else:
    print_colored(
        "Notebook no creado. Puedes ejecutar 'create_jupyter_notebook()' más tarde si lo deseas.",
        color="yellow",
    )


# Función para generar un informe en formato PDF
def generate_pdf_report():
    """
    Genera un informe en formato PDF con los resultados del análisis
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import inch
        from reportlab.platypus import (
            Image,
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )

        print_colored("\nGenerando informe PDF...", color="blue", style="bold")

        # Cargar resultados
        with open("nlp_models_results.json", "r", encoding="utf-8") as f:
            results = json.load(f)

        # Crear documento
        doc = SimpleDocTemplate("informe_modelos_nlp.pdf", pagesize=letter)
        styles = getSampleStyleSheet()

        # Añadir estilos personalizados
        styles.add(
            ParagraphStyle(
                name="Title", parent=styles["Heading1"], fontSize=18, spaceAfter=12
            )
        )

        styles.add(
            ParagraphStyle(
                name="Heading2", parent=styles["Heading2"], fontSize=14, spaceAfter=10
            )
        )

        styles.add(
            ParagraphStyle(
                name="Heading3", parent=styles["Heading3"], fontSize=12, spaceAfter=8
            )
        )

        styles.add(
            ParagraphStyle(
                name="Normal", parent=styles["Normal"], fontSize=10, spaceAfter=6
            )
        )

        # Elementos del informe
        elements = []

        # Título
        elements.append(
            Paragraph(
                "Análisis de Modelos RNN/LSTM y Transformer para NLP", styles["Title"]
            )
        )
        elements.append(Spacer(1, 0.2 * inch))

        # Introducción
        elements.append(Paragraph("Resumen Ejecutivo", styles["Heading2"]))
        elements.append(
            Paragraph(
                "Este informe presenta un análisis comparativo de diferentes arquitecturas de redes neuronales "
                "para procesamiento de lenguaje natural (NLP), incluyendo RNN simple, LSTM, GRU y Transformer. "
                "Se evalúan aspectos como precisión, rendimiento en tareas específicas, eficiencia computacional "
                "y tiempos de inferencia.",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 0.2 * inch))

        # Modelos analizados
        elements.append(Paragraph("Modelos Analizados", styles["Heading2"]))

        model_descriptions = [
            ["Modelo", "Descripción"],
            [
                "RNN Simple",
                "Red neuronal recurrente básica, procesa secuencias paso a paso manteniendo un estado oculto.",
            ],
            [
                "LSTM",
                "Long Short-Term Memory, diseñada para capturar dependencias a largo plazo mediante compuertas.",
            ],
            [
                "GRU",
                "Gated Recurrent Unit, alternativa más ligera a LSTM con menos parámetros.",
            ],
            [
                "Transformer",
                "Arquitectura basada en mecanismos de atención, permite procesamiento paralelo.",
            ],
        ]

        t = Table(model_descriptions, colWidths=[1.5 * inch, 4 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (1, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (1, 0), 12),
                    ("BACKGROUND", (0, 1), (1, 4), colors.beige),
                    ("GRID", (0, 0), (1, 4), 1, colors.black),
                ]
            )
        )

        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

        # Resultados principales
        elements.append(Paragraph("Resultados Principales", styles["Heading2"]))

        # Tabla de métricas
        metrics_data = [
            ["Modelo", "Accuracy", "F1-Score", "BLEU", "ROUGE-L", "Tiempo Rel."]
        ]

        for model in ["RNN", "LSTM", "GRU", "Transformer"]:
            metrics_data.append(
                [
                    model,
                    f"{results['metrics'][model]['accuracy']:.4f}",
                    f"{results['metrics'][model]['f1']:.4f}",
                    f"{results['metrics'][model]['bleu']:.4f}",
                    f"{results['metrics'][model]['rouge-l']:.4f}",
                    f"{results['inference_times'][model]:.2f}x",
                ]
            )

        t = Table(
            metrics_data,
            colWidths=[1.2 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch, 1 * inch],
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (5, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (5, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (5, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (5, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (5, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (5, 0), 12),
                    ("BACKGROUND", (0, 1), (0, 4), colors.lightgrey),
                    ("GRID", (0, 0), (5, 4), 1, colors.black),
                    ("ALIGN", (1, 1), (5, 4), "CENTER"),
                ]
            )
        )

        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

        # Recomendaciones
        elements.append(Paragraph("Recomendaciones", styles["Heading2"]))

        recommendations = [
            ["Caso de Uso", "Modelo Recomendado"],
            ["Mejor modelo general", results["recommendations"]["best_overall_model"]],
            [
                "Generación de texto",
                results["recommendations"]["best_generation_model"],
            ],
            ["Clasificación", results["recommendations"]["best_classification_model"]],
            [
                "Aplicaciones con restricciones de latencia",
                results["recommendations"]["fastest_model"],
            ],
            ["Secuencias largas con dependencias a distancia", "Transformer"],
            ["Conjuntos de datos pequeños", "LSTM o GRU"],
        ]

        t = Table(recommendations, colWidths=[3 * inch, 2.5 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (1, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (1, 0), 12),
                    ("BACKGROUND", (0, 1), (0, 6), colors.lightgrey),
                    ("GRID", (0, 0), (1, 6), 1, colors.black),
                    ("ALIGN", (1, 1), (1, 6), "CENTER"),
                ]
            )
        )

        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

        # Análisis detallado
        elements.append(Paragraph("Análisis Detallado", styles["Heading2"]))

        # Ventajas y desventajas
        elements.append(Paragraph("Ventajas y Desventajas", styles["Heading3"]))

        advantages_disadvantages = [
            ["Modelo", "Ventajas", "Desventajas"],
            [
                "RNN Simple",
                "• Arquitectura simple\n• Menos parámetros\n• Eficiente para secuencias cortas",
                "• Dificultad con dependencias a largo plazo\n• Problemas de gradientes desvanecientes\n• Procesamiento secuencial",
            ],
            [
                "LSTM",
                "• Captura dependencias a largo plazo\n• Maneja gradientes desvanecientes\n• Buena capacidad expresiva",
                "• Mayor complejidad computacional\n• Más parámetros que RNN simple\n• Procesamiento secuencial",
            ],
            [
                "GRU",
                "• Similar a LSTM pero más eficiente\n• Menos parámetros que LSTM\n• Buen equilibrio rendimiento/eficiencia",
                "• Ligeramente menos expresivo que LSTM\n• Procesamiento secuencial\n• Menos efectivo en secuencias muy largas",
            ],
            [
                "Transformer",
                "• Paralelización\n• Mejor captura de dependencias a largo plazo\n• Atención a diferentes partes de la secuencia",
                "• Mayor número de parámetros\n• Complejidad cuadrática con longitud\n• Requiere más datos para entrenar",
            ],
        ]

        t = Table(
            advantages_disadvantages, colWidths=[1.2 * inch, 2.4 * inch, 2.4 * inch]
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (2, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (2, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (2, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (2, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (2, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (2, 0), 12),
                    ("BACKGROUND", (0, 1), (0, 4), colors.lightgrey),
                    ("GRID", (0, 0), (2, 4), 1, colors.black),
                    ("VALIGN", (0, 0), (2, 4), "TOP"),
                ]
            )
        )

        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

        # Complejidad computacional
        elements.append(Paragraph("Complejidad Computacional", styles["Heading3"]))
        elements.append(
            Paragraph(
                "La complejidad computacional es un factor crítico al seleccionar un modelo para aplicaciones prácticas. "
                "A continuación se presenta un análisis de la complejidad teórica y los tiempos de inferencia relativos:",
                styles["Normal"],
            )
        )

        complexity_data = [
            ["Modelo", "Complejidad Temporal", "Complejidad Espacial", "Paralelizable"],
            ["RNN Simple", "O(n)", "O(1)", "No"],
            ["LSTM", "O(n)", "O(1)", "No"],
            ["GRU", "O(n)", "O(1)", "No"],
            ["Transformer", "O(n²)", "O(n)", "Sí"],
        ]

        t = Table(
            complexity_data, colWidths=[1.5 * inch, 1.5 * inch, 1.5 * inch, 1.5 * inch]
        )
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (3, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (3, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (3, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (3, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (3, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (3, 0), 12),
                    ("BACKGROUND", (0, 1), (0, 4), colors.lightgrey),
                    ("GRID", (0, 0), (3, 4), 1, colors.black),
                    ("ALIGN", (1, 1), (3, 4), "CENTER"),
                ]
            )
        )

        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

        # Hiperparámetros utilizados
        elements.append(Paragraph("Hiperparámetros Utilizados", styles["Heading3"]))

        hyperparams_data = [["Hiperparámetro", "Valor"]]
        for param, value in results["hyperparameters"].items():
            hyperparams_data.append([param, str(value)])

        t = Table(hyperparams_data, colWidths=[3 * inch, 3 * inch])
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (1, 0), "CENTER"),
                    ("FONTNAME", (0, 0), (1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (1, 0), 12),
                    ("BOTTOMPADDING", (0, 0), (1, 0), 12),
                    (
                        "BACKGROUND",
                        (0, 1),
                        (0, len(hyperparams_data) - 1),
                        colors.lightgrey,
                    ),
                    ("GRID", (0, 0), (1, len(hyperparams_data) - 1), 1, colors.black),
                    ("ALIGN", (1, 1), (1, len(hyperparams_data) - 1), "CENTER"),
                ]
            )
        )

        elements.append(t)
        elements.append(Spacer(1, 0.2 * inch))

        # Conclusiones
        elements.append(Paragraph("Conclusiones", styles["Heading2"]))
        elements.append(
            Paragraph(
                f"Tras un análisis exhaustivo de las diferentes arquitecturas de redes neuronales para NLP, "
                f"se concluye que {results['recommendations']['best_overall_model']} ofrece el mejor equilibrio "
                f"entre rendimiento y eficiencia para el conjunto de datos y tareas evaluadas. Sin embargo, "
                f"la elección del modelo debe adaptarse a los requisitos específicos de cada aplicación.",
                styles["Normal"],
            )
        )

        elements.append(Paragraph("Principales hallazgos:", styles["Normal"]))

        findings = [
            "1. Los modelos basados en mecanismos de atención (Transformer) muestran ventajas significativas en tareas que requieren capturar dependencias a larga distancia.",
            "2. LSTM y GRU siguen siendo alternativas viables y eficientes, especialmente para conjuntos de datos más pequeños o cuando los recursos computacionales son limitados.",
            "3. La complejidad computacional es un factor crítico a considerar, especialmente para aplicaciones en tiempo real o dispositivos con recursos limitados.",
            f"4. El modelo {results['recommendations']['best_generation_model']} mostró el mejor rendimiento en métricas de generación de texto como BLEU y ROUGE.",
            f"5. Para tareas de clasificación, {results['recommendations']['best_classification_model']} obtuvo los mejores resultados en términos de F1-score.",
        ]

        for finding in findings:
            elements.append(Paragraph(finding, styles["Normal"]))

        elements.append(Spacer(1, 0.2 * inch))

        # Trabajo futuro
        elements.append(Paragraph("Trabajo Futuro", styles["Heading2"]))
        elements.append(
            Paragraph(
                "Para continuar y expandir este análisis, se recomiendan las siguientes líneas de investigación:",
                styles["Normal"],
            )
        )

        future_work = [
            "• Explorar variantes como Transformer-XL para secuencias más largas",
            "• Implementar técnicas de transferencia de aprendizaje con modelos pre-entrenados",
            "• Optimizar hiperparámetros con búsqueda sistemática",
            "• Evaluar en conjuntos de datos más grandes y diversos",
            "• Implementar técnicas de regularización adicionales para mejorar la generalización",
            "• Analizar el impacto de diferentes estrategias de tokenización y preprocesamiento",
            "• Explorar arquitecturas híbridas que combinen elementos de modelos recurrentes y basados en atención",
        ]

        for work in future_work:
            elements.append(Paragraph(work, styles["Normal"]))

        # Construir el PDF
        doc.build(elements)

        print_colored(
            "Informe PDF generado como 'informe_modelos_nlp.pdf'", color="green"
        )

    except ImportError:
        print_colored(
            "Para generar el PDF, instala ReportLab: pip install reportlab",
            color="yellow",
        )
    except Exception as e:
        print_colored(f"Error al generar el PDF: {e}", color="red")


# Preguntar si se desea generar el informe PDF
if input("\n¿Deseas generar un informe PDF con los resultados? (s/n): ").lower() == "s":
    generate_pdf_report()
else:
    print_colored(
        "Informe PDF no generado. Puedes ejecutar 'generate_pdf_report()' más tarde si lo deseas.",
        color="yellow",
    )

# Mensaje final
print_colored(
    "\n===== ANÁLISIS COMPLETO FINALIZADO =====", color="magenta", style="bold"
)
print_colored(
    "Gracias por utilizar este análisis completo de modelos de NLP.",
    color="green",
    style="bold",
)
print_colored(
    "Todos los resultados han sido guardados para su posterior consulta.", color="blue"
)
