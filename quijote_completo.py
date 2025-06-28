# Celda 1: Importación de librerías y configuración inicial
# Importación de librerías necesarias
import os
import re
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from colorama import Fore, Style, init
from matplotlib.ticker import MaxNLocator
from scipy.stats import entropy
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, Dataset

# Inicializar colorama para colores en la terminal
init(autoreset=True)

# Verificar si CUDA está disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{Fore.CYAN}Utilizando dispositivo: {Fore.YELLOW}{device}{Style.RESET_ALL}")

# Configuración de visualización
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 12


# Celda 2: Clase para procesar y preparar el texto
class TextProcessor:
    def __init__(self, file_path):
        """
        Inicializa el procesador de texto
        Args:
            file_path: Ruta al archivo de texto
        """
        print(f"{Fore.GREEN}Inicializando procesador de texto...{Style.RESET_ALL}")
        self.file_path = file_path
        self.text = self.load_text()
        self.chars = sorted(list(set(self.text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        print(
            f"{Fore.GREEN}Procesador de texto inicializado correctamente.{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}Tamaño del vocabulario: {Fore.WHITE}{self.vocab_size} caracteres únicos"
        )

    def load_text(self):
        """
        Carga y limpia el texto del archivo
        Returns:
            str: Texto limpio
        """
        try:
            print(f"{Fore.BLUE}Cargando texto desde: {self.file_path}{Style.RESET_ALL}")
            with open(self.file_path, "r", encoding="utf-8") as f:
                text = f.read()

            # Limpieza básica del texto
            text = text.lower()
            # Eliminar caracteres no deseados si es necesario
            # text = re.sub(r'[^\w\s]', '', text)

            print(
                f"{Fore.GREEN}Texto cargado correctamente. Longitud: {len(text):,} caracteres{Style.RESET_ALL}"
            )
            return text
        except Exception as e:
            print(f"{Fore.RED}Error al cargar el archivo: {e}{Style.RESET_ALL}")
            return ""

    def get_stats(self):
        """
        Obtiene estadísticas básicas del texto
        Returns:
            dict: Diccionario con estadísticas
        """
        print(f"{Fore.BLUE}Calculando estadísticas del texto...{Style.RESET_ALL}")
        words = self.text.split()
        word_count = len(words)
        unique_words = len(set(words))
        char_count = len(self.text)

        # Palabras más comunes
        word_counter = Counter(words)
        most_common_words = word_counter.most_common(10)

        # Longitud de palabras
        word_lengths = [len(word) for word in words]
        avg_word_length = sum(word_lengths) / len(word_lengths)

        # Frecuencia de caracteres
        char_counter = Counter(self.text)
        most_common_chars = char_counter.most_common(10)

        # Calcular n-gramas
        bigrams = ["".join(self.text[i : i + 2]) for i in range(len(self.text) - 1)]
        trigrams = ["".join(self.text[i : i + 3]) for i in range(len(self.text) - 2)]

        bigram_counter = Counter(bigrams)
        trigram_counter = Counter(trigrams)

        return {
            "total_chars": char_count,
            "vocab_size": self.vocab_size,
            "total_words": word_count,
            "unique_words": unique_words,
            "most_common_words": most_common_words,
            "avg_word_length": avg_word_length,
            "most_common_chars": most_common_chars,
            "word_lengths": word_lengths,
            "word_counter": word_counter,
            "most_common_bigrams": bigram_counter.most_common(10),
            "most_common_trigrams": trigram_counter.most_common(10),
        }

    def create_sequences(self, seq_length, max_sequences=None):
        """
        Crea secuencias de entrenamiento de longitud fija
        Args:
            seq_length: Longitud de cada secuencia
            max_sequences: Número máximo de secuencias a crear (None para todas)
        Returns:
            tuple: (X, y) donde X son las secuencias de entrada y y son los objetivos
        """
        print(
            f"{Fore.BLUE}Creando secuencias de entrenamiento con longitud {seq_length}...{Style.RESET_ALL}"
        )
        sequences = []
        next_chars = []

        for i in range(0, len(self.text) - seq_length):
            sequences.append(self.text[i : i + seq_length])
            next_chars.append(self.text[i + seq_length])

            # Limitar el número de secuencias si se especifica
            if max_sequences is not None and len(sequences) >= max_sequences:
                break

        print(
            f"{Fore.GREEN}Se han creado {len(sequences):,} secuencias de entrenamiento.{Style.RESET_ALL}"
        )

        # Convertir a índices
        print(
            f"{Fore.BLUE}Codificando secuencias en formato one-hot...{Style.RESET_ALL}"
        )
        X = np.zeros((len(sequences), seq_length, self.vocab_size), dtype=np.bool_)
        y = np.zeros((len(sequences), self.vocab_size), dtype=np.bool_)

        for i, sequence in enumerate(sequences):
            for t, char in enumerate(sequence):
                X[i, t, self.char_to_idx[char]] = 1
            y[i, self.char_to_idx[next_chars[i]]] = 1

        print(
            f"{Fore.GREEN}Codificación completada. Forma de X: {X.shape}, Forma de y: {y.shape}{Style.RESET_ALL}"
        )
        print(
            f"{Fore.YELLOW}Memoria aproximada utilizada: {X.nbytes / (1024**2):.2f} MB + {y.nbytes / (1024**2):.2f} MB = {(X.nbytes + y.nbytes) / (1024**2):.2f} MB{Style.RESET_ALL}"
        )

        return torch.FloatTensor(X), torch.FloatTensor(y), sequences, next_chars


# Celda 3: Clase de conjunto de datos y modelo LSTM
# Clase de conjunto de datos personalizado para PyTorch
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# Modelo LSTM para generación de texto
class TextGenerationModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        """
        Inicializa el modelo de generación de texto
        Args:
            input_size: Tamaño del vocabulario
            hidden_size: Tamaño de las capas ocultas
            num_layers: Número de capas LSTM
            output_size: Tamaño de la salida (igual al tamaño del vocabulario)
            dropout: Tasa de dropout
        """
        super(TextGenerationModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, hidden=None):
        """
        Forward pass del modelo
        Args:
            x: Tensor de entrada
            hidden: Estado oculto inicial (opcional)
        Returns:
            tuple: (output, hidden_state)
        """
        batch_size = x.size(0)

        # Inicializar estado oculto si no se proporciona
        if hidden is None:
            h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
            hidden = (h0, c0)

        # Forward pass a través de LSTM
        out, hidden = self.lstm(x, hidden)

        # Aplicar dropout
        out = self.dropout(out[:, -1, :])

        # Capa completamente conectada
        out = self.fc(out)

        # Aplicar softmax
        out = self.softmax(out)

        return out, hidden

    def get_internal_states(self, x):
        """
        Obtiene los estados internos de la LSTM para visualización
        Args:
            x: Tensor de entrada
        Returns:
            dict: Diccionario con estados internos
        """
        batch_size = x.size(0)

        # Inicializar estado oculto
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        hidden = (h0, c0)

        # Forward pass a través de LSTM
        out, (h_n, c_n) = self.lstm(x, hidden)

        return {
            "output": out.detach().cpu().numpy(),
            "hidden_state": h_n.detach().cpu().numpy(),
            "cell_state": c_n.detach().cpu().numpy(),
        }

    def generate_text(self, processor, seed_text, length=100, temperature=1.0):
        """
        Genera texto a partir de una semilla
        Args:
            processor: Procesador de texto
            seed_text: Texto semilla para comenzar la generación
            length: Longitud del texto a generar
            temperature: Controla la aleatoriedad (menor = más determinista)
        Returns:
            str: Texto generado
        """
        print(
            f"{Fore.BLUE}Generando texto a partir de la semilla: '{seed_text}'{Style.RESET_ALL}"
        )
        print(
            f"{Fore.BLUE}Temperatura: {temperature} (menor = más determinista, mayor = más aleatorio){Style.RESET_ALL}"
        )

        self.eval()
        generated_text = seed_text

        # Asegurarse de que la semilla tenga la longitud correcta
        seed_text = seed_text[-len(seed_text) :]

        # Mostrar barra de progreso
        print(f"{Fore.YELLOW}Progreso de generación:{Style.RESET_ALL}")
        progress_interval = max(1, length // 20)

        # Guardar historial de probabilidades para análisis
        probability_history = []

        with torch.no_grad():
            for i in range(length):
                # Actualizar barra de progreso
                if i % progress_interval == 0 or i == length - 1:
                    progress = int((i + 1) / length * 20)
                    print(
                        f"\r[{'=' * progress}{' ' * (20 - progress)}] {(i + 1) / length * 100:.1f}%",
                        end="",
                    )

                # Preparar la entrada
                x = np.zeros((1, len(seed_text), processor.vocab_size))
                for t, char in enumerate(seed_text):
                    if char in processor.char_to_idx:
                        x[0, t, processor.char_to_idx[char]] = 1

                x = torch.FloatTensor(x).to(device)

                # Predecir el siguiente carácter
                prediction, _ = self(x)

                # Guardar probabilidades originales
                original_probs = prediction.cpu().numpy()[0]

                # Aplicar temperatura
                prediction = prediction.cpu().numpy()[0]
                prediction = np.log(prediction) / temperature
                exp_prediction = np.exp(prediction)
                prediction = exp_prediction / np.sum(exp_prediction)

                # Guardar historial de probabilidades
                probability_history.append(
                    {
                        "original": original_probs,
                        "with_temp": prediction,
                        "selected_index": np.argmax(prediction),
                        "entropy": entropy(prediction),
                    }
                )

                # Muestrear el siguiente carácter
                next_index = np.random.choice(len(prediction), p=prediction)
                next_char = processor.idx_to_char[next_index]

                # Actualizar el texto generado y la semilla
                generated_text += next_char
                seed_text = seed_text[1:] + next_char

        print()  # Nueva línea después de la barra de progreso
        print(f"{Fore.GREEN}Generación de texto completada.{Style.RESET_ALL}")

        # Calcular entropía promedio (medida de incertidumbre/aleatoriedad)
        avg_entropy = np.mean([p["entropy"] for p in probability_history])
        print(
            f"{Fore.YELLOW}Entropía promedio durante la generación: {avg_entropy:.4f}{Style.RESET_ALL}"
        )

        return generated_text, probability_history


# Celda 4: Funciones de entrenamiento y visualización
# Función para entrenar el modelo
def train_model(
    model, dataloader, epochs, learning_rate=0.001, processor=None, seed_text=None
):
    """
    Entrena el modelo
    Args:
        model: Modelo a entrenar
        dataloader: DataLoader con los datos de entrenamiento
        epochs: Número de épocas
        learning_rate: Tasa de aprendizaje
        processor: Procesador de texto (opcional, para generar muestras durante el entrenamiento)
        seed_text: Texto semilla para generar muestras (opcional)
    Returns:
        dict: Historial de entrenamiento
    """
    print(f"{Fore.BLUE}Configurando entrenamiento...{Style.RESET_ALL}")
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    history = {
        "loss": [],
        "perplexity": [],
        "time_per_epoch": [],
        "samples": {} if processor and seed_text else None,
    }

    total_batches = len(dataloader)
    print(
        f"{Fore.GREEN}Entrenamiento configurado. Total de batches por época: {total_batches}{Style.RESET_ALL}"
    )

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0

        # Mostrar barra de progreso
        print(f"{Fore.YELLOW}Época {epoch+1}/{epochs}:{Style.RESET_ALL}")
        progress_interval = max(1, total_batches // 20)

        for batch_idx, (X_batch, y_batch) in enumerate(dataloader):
            # Actualizar barra de progreso
            if batch_idx % progress_interval == 0 or batch_idx == total_batches - 1:
                progress = int((batch_idx + 1) / total_batches * 20)
                print(
                    f"\r[{'=' * progress}{' ' * (20 - progress)}] {(batch_idx + 1) / total_batches * 100:.1f}%",
                    end="",
                )

            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            outputs, _ = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        # Nueva línea después de la barra de progreso
        print()

        # Calcular estadísticas de la época
        avg_loss = epoch_loss / total_batches
        perplexity = np.exp(avg_loss)
        epoch_time = time.time() - epoch_start_time

        # Actualizar historial
        history["loss"].append(avg_loss)
        history["perplexity"].append(perplexity)
        history["time_per_epoch"].append(epoch_time)

        print(
            f"{Fore.GREEN}Época {epoch+1}/{epochs} completada en {epoch_time:.2f} segundos. Pérdida: {avg_loss:.4f}, Perplejidad: {perplexity:.4f}{Style.RESET_ALL}"
        )

        # Generar muestra de texto si se proporcionan processor y seed_text
        if processor and seed_text and (epoch + 1) % max(1, epochs // 5) == 0:
            print(
                f"{Fore.CYAN}Generando muestra de texto en la época {epoch+1}...{Style.RESET_ALL}"
            )
            model.eval()
            sample, _ = model.generate_text(
                processor, seed_text, length=100, temperature=1.0
            )
            history["samples"][epoch + 1] = sample
            print(f"{Fore.CYAN}Muestra: {sample[:100]}...{Style.RESET_ALL}")
            model.train()

    print(
        f"{Fore.GREEN}Entrenamiento completado en {sum(history['time_per_epoch']):.2f} segundos.{Style.RESET_ALL}"
    )
    return history


# Función para visualizar la distribución de probabilidades
def visualize_prediction_distribution(
    model, processor, seed_text, temperatures=[0.5, 1.0, 1.5]
):
    """
    Visualiza la distribución de probabilidades para diferentes temperaturas
    Args:
        model: Modelo entrenado
        processor: Procesador de texto
        seed_text: Texto semilla
        temperatures: Lista de temperaturas a visualizar
    """
    print(
        f"{Fore.BLUE}Visualizando distribución de probabilidades para la semilla: '{seed_text}'{Style.RESET_ALL}"
    )

    plt.figure(figsize=(15, 10))

    # Preparar la entrada
    x = np.zeros((1, len(seed_text), processor.vocab_size))
    for t, char in enumerate(seed_text):
        if char in processor.char_to_idx:
            x[0, t, processor.char_to_idx[char]] = 1

    x = torch.FloatTensor(x).to(device)

    # Obtener predicciones
    with torch.no_grad():
        prediction, _ = model(x)
        prediction = prediction.cpu().numpy()[0]

    # Mostrar top 15 caracteres más probables
    top_indices = np.argsort(prediction)[-15:][::-1]
    chars = [processor.idx_to_char[idx] for idx in top_indices]
    probs = [prediction[idx] for idx in top_indices]

    # Crear subplots para diferentes temperaturas
    for i, temp in enumerate(temperatures):
        plt.subplot(len(temperatures), 1, i + 1)

        # Aplicar temperatura
        scaled_prediction = np.log(prediction) / temp
        exp_prediction = np.exp(scaled_prediction)
        scaled_prediction = exp_prediction / np.sum(exp_prediction)

        # Obtener probabilidades ajustadas
        scaled_probs = [scaled_prediction[idx] for idx in top_indices]

        # Crear gráfico de barras
        bars = plt.bar(chars, scaled_probs, color="skyblue")
        plt.title(f"Distribución de probabilidades (Temperatura = {temp})")
        plt.ylabel("Probabilidad")
        plt.xlabel("Caracteres")

        # Añadir valores
        for bar, prob in zip(bars, scaled_probs):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{prob:.3f}",
                ha="center",
                va="bottom",
                rotation=0,
            )

    plt.tight_layout()
    plt.savefig("distribucion_probabilidades.png")
    plt.show()

    print(
        f"{Fore.GREEN}Visualización de distribución de probabilidades completada.{Style.RESET_ALL}"
    )


# Función para visualizar los estados internos de la LSTM
def visualize_lstm_states(model, processor, text, seq_length):
    """
    Visualiza los estados internos de la LSTM para un texto dado
    Args:
        model: Modelo entrenado
        processor: Procesador de texto
        text: Texto de entrada
        seq_length: Longitud de la secuencia
    """
    print(f"{Fore.BLUE}Visualizando estados internos de la LSTM...{Style.RESET_ALL}")

    if len(text) < seq_length:
        text = text.ljust(seq_length)

    # Preparar la entrada
    x = np.zeros((1, seq_length, processor.vocab_size))
    for t, char in enumerate(text[:seq_length]):
        if char in processor.char_to_idx:
            x[0, t, processor.char_to_idx[char]] = 1

    x = torch.FloatTensor(x).to(device)

    # Obtener estados internos
    states = model.get_internal_states(x)

    # Visualizar estados ocultos finales para cada capa
    plt.figure(figsize=(15, 5 * model.num_layers))

    for layer in range(model.num_layers):
        plt.subplot(model.num_layers, 1, layer + 1)

        # Obtener estado oculto para esta capa
        hidden_state = states["hidden_state"][layer, 0]

        # Crear heatmap
        sns.heatmap(
            hidden_state.reshape(1, -1),
            cmap="viridis",
            xticklabels=False,
            yticklabels=False,
        )
        plt.title(f"Estado oculto - Capa LSTM {layer+1}")
        plt.ylabel("Neurona")

    plt.tight_layout()
    plt.savefig("lstm_estados_internos.png")
    plt.show()

    # Visualizar la evolución de la activación a lo largo de la secuencia
    plt.figure(figsize=(15, 6))

    # Tomar una muestra de neuronas para visualizar (primeras 10)
    output = states["output"][0, :, :10]  # [seq_length, 10]

    sns.heatmap(
        output,
        cmap="coolwarm",
        xticklabels=range(1, 11),  # 10 neuronas
        yticklabels=list(text[:seq_length]),
    )
    plt.title("Activación de neuronas a lo largo de la secuencia")
    plt.xlabel("Índice de neurona")
    plt.ylabel("Carácter de entrada")

    plt.tight_layout()
    plt.savefig("lstm_activacion_secuencia.png")
    plt.show()

    print(f"{Fore.GREEN}Visualización de estados internos completada.{Style.RESET_ALL}")


# Función para analizar n-gramas
def analyze_ngrams(original_text, generated_text, n=2, top_k=20):
    """
    Compara los n-gramas más comunes entre el texto original y el generado
    Args:
        original_text: Texto original
        generated_text: Texto generado
        n: Tamaño del n-grama
        top_k: Número de n-gramas más comunes a mostrar
    Returns:
        float: Similitud de Jaccard entre conjuntos de n-gramas
    """
    print(
        f"{Fore.BLUE}Analizando {n}-gramas entre texto original y generado...{Style.RESET_ALL}"
    )

    def get_ngrams(text, n):
        ngrams = []
        for i in range(len(text) - n + 1):
            ngrams.append(text[i : i + n])
        return Counter(ngrams)

    # Obtener n-gramas
    original_ngrams = get_ngrams(original_text, n)
    generated_ngrams = get_ngrams(generated_text, n)

    # Top k n-gramas
    top_original = original_ngrams.most_common(top_k)
    top_generated = generated_ngrams.most_common(top_k)

    # Visualizar
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Original
    ngrams_orig, counts_orig = zip(*top_original)
    ngrams_orig_display = [
        f"'{ng}'" if not ng.isspace() else "'espacio'" for ng in ngrams_orig
    ]
    ax1.barh(ngrams_orig_display, counts_orig, color="blue")
    ax1.set_title(f"Top {top_k} {n}-gramas en texto original")
    ax1.set_xlabel("Frecuencia")

    # Generado
    ngrams_gen, counts_gen = zip(*top_generated)
    ngrams_gen_display = [
        f"'{ng}'" if not ng.isspace() else "'espacio'" for ng in ngrams_gen
    ]
    ax2.barh(ngrams_gen_display, counts_gen, color="green")
    ax2.set_title(f"Top {top_k} {n}-gramas en texto generado")
    ax2.set_xlabel("Frecuencia")

    plt.tight_layout()
    plt.savefig(f"comparacion_{n}gramas.png")
    plt.show()

    # Calcular similitud de Jaccard entre conjuntos de n-gramas
    original_set = set(original_ngrams.keys())
    generated_set = set(generated_ngrams.keys())

    intersection = len(original_set.intersection(generated_set))
    union = len(original_set.union(generated_set))

    jaccard = intersection / union if union > 0 else 0

    print(
        f"{Fore.CYAN}Similitud de Jaccard para {n}-gramas: {jaccard:.4f}{Style.RESET_ALL}"
    )
    print(
        f"Interpretación: {jaccard:.2%} de los {n}-gramas son compartidos entre ambos textos"
    )

    # Calcular n-gramas comunes y exclusivos
    common_ngrams = original_set.intersection(generated_set)
    only_original = original_set - generated_set
    only_generated = generated_set - original_set

    print(f"{Fore.YELLOW}N-gramas comunes: {len(common_ngrams)}")
    print(f"{Fore.YELLOW}N-gramas solo en original: {len(only_original)}")
    print(f"{Fore.YELLOW}N-gramas solo en generado: {len(only_generated)}")

    return jaccard


# Función para evaluar la calidad del texto generado
def evaluate_text_quality(original_text, generated_text):
    """
    Evalúa la calidad del texto generado con múltiples métricas
    Args:
        original_text: Texto original
        generated_text: Texto generado
    Returns:
        dict: Diccionario con métricas de calidad
    """
    print(f"\n{Fore.CYAN}EVALUACIÓN DE CALIDAD DEL TEXTO GENERADO{Style.RESET_ALL}")

    # 1. Distribución de longitud de palabras
    orig_words = original_text.split()
    gen_words = generated_text.split()

    orig_lengths = [len(word) for word in orig_words]
    gen_lengths = [len(word) for word in gen_words]

    plt.figure(figsize=(12, 6))
    plt.hist(orig_lengths, alpha=0.5, label="Original", bins=15, color="blue")
    plt.hist(gen_lengths, alpha=0.5, label="Generado", bins=15, color="green")
    plt.title("Comparación de distribución de longitud de palabras")
    plt.xlabel("Longitud de palabra")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparacion_longitud_palabras.png")
    plt.show()

    # 2. Frecuencia de caracteres
    orig_chars = Counter(original_text)
    gen_chars = Counter(generated_text)

    # Normalizar por longitud total
    orig_freq = {k: v / len(original_text) for k, v in orig_chars.items()}
    gen_freq = {k: v / len(generated_text) for k, v in gen_chars.items()}

    # Seleccionar caracteres comunes para comparar
    common_chars = set([k for k, v in orig_chars.most_common(20)]).union(
        set([k for k, v in gen_chars.most_common(20)])
    )

    # Filtrar espacios y caracteres no imprimibles para visualización
    common_chars = [c for c in common_chars if c.isprintable() and not c.isspace()][:15]

    # Crear gráfico de comparación
    plt.figure(figsize=(15, 6))

    x = np.arange(len(common_chars))
    width = 0.35

    orig_values = [orig_freq.get(c, 0) for c in common_chars]
    gen_values = [gen_freq.get(c, 0) for c in common_chars]

    plt.bar(x - width / 2, orig_values, width, label="Original", color="blue")
    plt.bar(x + width / 2, gen_values, width, label="Generado", color="green")

    plt.xlabel("Caracteres")
    plt.ylabel("Frecuencia relativa")
    plt.title("Comparación de frecuencia de caracteres")
    plt.xticks(x, common_chars)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("comparacion_frecuencia_caracteres.png")
    plt.show()

    # 3. Calcular KL divergence entre distribuciones de caracteres
    all_chars = set(orig_chars.keys()).union(set(gen_chars.keys()))

    # Crear distribuciones completas
    p = np.array([orig_freq.get(c, 1e-10) for c in all_chars])
    q = np.array([gen_freq.get(c, 1e-10) for c in all_chars])

    # Normalizar
    p = p / p.sum()
    q = q / q.sum()

    # Calcular KL divergence
    kl_div = np.sum(p * np.log(p / q))

    print(
        f"{Fore.YELLOW}KL Divergence entre distribuciones de caracteres: {kl_div:.4f}{Style.RESET_ALL}"
    )
    print(f"Interpretación: Menor valor indica mayor similitud entre distribuciones")

    # 4. Comparar palabras más comunes
    orig_word_counter = Counter(orig_words)
    gen_word_counter = Counter(gen_words)

    top_orig_words = [word for word, _ in orig_word_counter.most_common(10)]
    top_gen_words = [word for word, _ in gen_word_counter.most_common(10)]

    # Calcular solapamiento
    overlap = len(set(top_orig_words).intersection(set(top_gen_words)))

    print(
        f"{Fore.YELLOW}Solapamiento de palabras más comunes: {overlap}/10 ({overlap/10:.0%}){Style.RESET_ALL}"
    )

    # Visualizar palabras más comunes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Original
    words_orig, counts_orig = zip(*orig_word_counter.most_common(10))
    ax1.barh(words_orig, counts_orig, color="blue")
    ax1.set_title("Top 10 palabras más comunes en texto original")
    ax1.set_xlabel("Frecuencia")

    # Generado
    words_gen, counts_gen = zip(*gen_word_counter.most_common(10))
    ax2.barh(words_gen, counts_gen, color="green")
    ax2.set_title("Top 10 palabras más comunes en texto generado")
    ax2.set_xlabel("Frecuencia")

    plt.tight_layout()
    plt.savefig("comparacion_palabras_comunes.png")
    plt.show()

    # Calcular correlación de longitud de palabras
    # Asegurarse de que ambos arrays tengan la misma longitud
    min_length = min(len(orig_lengths), len(gen_lengths))
    if min_length > 0:
        # Tomar solo los primeros min_length elementos de cada array
        correlation = np.corrcoef(orig_lengths[:min_length], gen_lengths[:min_length])[
            0, 1
        ]
    else:
        correlation = 0.0

    return {
        "kl_divergence": kl_div,
        "word_overlap": overlap / 10,
        "word_length_correlation": correlation,
    }


# Función para visualizar la evolución de la perplejidad
def plot_training_metrics(history):
    """
    Visualiza métricas de entrenamiento como pérdida y perplejidad
    Args:
        history: Historial de entrenamiento
    """
    # Crear figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Gráfico de pérdida
    epochs = range(1, len(history["loss"]) + 1)
    ax1.plot(epochs, history["loss"], "b-o", linewidth=2, markersize=8)
    ax1.set_title("Evolución de la pérdida durante el entrenamiento")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Pérdida")
    ax1.grid(True, alpha=0.3)

    # Añadir valores como etiquetas
    for i, loss in enumerate(history["loss"]):
        ax1.annotate(
            f"{loss:.4f}",
            (i + 1, loss),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    # Gráfico de perplejidad
    ax2.plot(epochs, history["perplexity"], "r-o", linewidth=2, markersize=8)
    ax2.set_title("Evolución de la perplejidad durante el entrenamiento")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Perplejidad (menor es mejor)")
    ax2.grid(True, alpha=0.3)

    # Añadir valores como etiquetas
    for i, perp in enumerate(history["perplexity"]):
        ax2.annotate(
            f"{perp:.2f}",
            (i + 1, perp),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.tight_layout()
    plt.savefig("metricas_entrenamiento.png")
    plt.show()

    # Gráfico de tiempo por época
    plt.figure(figsize=(12, 6))
    plt.bar(epochs, history["time_per_epoch"], color="green", alpha=0.7)
    plt.title("Tiempo de entrenamiento por época")
    plt.xlabel("Época")
    plt.ylabel("Tiempo (segundos)")
    plt.grid(True, axis="y", alpha=0.3)

    # Añadir valores como etiquetas
    for i, time_val in enumerate(history["time_per_epoch"]):
        plt.annotate(
            f"{time_val:.2f}s",
            (i + 1, time_val),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.savefig("tiempo_entrenamiento.png")
    plt.show()

    # Si hay muestras generadas durante el entrenamiento, mostrarlas
    if history["samples"]:
        print(
            f"\n{Fore.CYAN}EVOLUCIÓN DE MUESTRAS GENERADAS DURANTE EL ENTRENAMIENTO{Style.RESET_ALL}"
        )
        for epoch, sample in history["samples"].items():
            print(f"{Fore.YELLOW}Época {epoch}:{Style.RESET_ALL} {sample[:100]}...")


# Función para visualizar la entropía durante la generación de texto
def plot_generation_entropy(probability_history, processor):
    """
    Visualiza la entropía y otras métricas durante la generación de texto
    Args:
        probability_history: Historial de probabilidades durante la generación
        processor: Procesador de texto
    """
    # Extraer entropías
    entropies = [p["entropy"] for p in probability_history]

    # Crear gráfico de entropía
    plt.figure(figsize=(12, 6))
    plt.plot(entropies, color="purple")
    plt.title("Entropía durante la generación de texto")
    plt.xlabel("Paso de generación")
    plt.ylabel("Entropía (mayor = más incertidumbre)")
    plt.grid(True, alpha=0.3)

    # Añadir línea de entropía promedio
    avg_entropy = np.mean(entropies)
    plt.axhline(
        y=avg_entropy,
        color="r",
        linestyle="--",
        label=f"Entropía promedio: {avg_entropy:.4f}",
    )
    plt.legend()

    plt.savefig("entropia_generacion.png")
    plt.show()

    # Visualizar la confianza del modelo (probabilidad máxima)
    max_probs = [np.max(p["with_temp"]) for p in probability_history]

    plt.figure(figsize=(12, 6))
    plt.plot(max_probs, color="green")
    plt.title("Confianza del modelo durante la generación de texto")
    plt.xlabel("Paso de generación")
    plt.ylabel("Probabilidad máxima")
    plt.grid(True, alpha=0.3)

    # Añadir línea de confianza promedio
    avg_confidence = np.mean(max_probs)
    plt.axhline(
        y=avg_confidence,
        color="r",
        linestyle="--",
        label=f"Confianza promedio: {avg_confidence:.4f}",
    )
    plt.legend()

    plt.savefig("confianza_generacion.png")
    plt.show()

    # Visualizar distribución de caracteres seleccionados
    selected_indices = [p["selected_index"] for p in probability_history]
    selected_chars = [processor.idx_to_char[idx] for idx in selected_indices]

    # Contar frecuencias
    char_counter = Counter(selected_chars)

    # Mostrar los 15 caracteres más comunes
    most_common = char_counter.most_common(15)
    chars, counts = zip(*most_common)

    plt.figure(figsize=(12, 6))
    plt.bar(chars, counts, color="skyblue")
    plt.title("Caracteres más frecuentemente seleccionados durante la generación")
    plt.xlabel("Carácter")
    plt.ylabel("Frecuencia")
    plt.grid(True, axis="y", alpha=0.3)

    plt.savefig("caracteres_seleccionados.png")
    plt.show()


# Celda 5: Función principal y ejecución
# Función principal
def main():
    # Ruta al archivo de texto
    file_path = "don-quijote-de-la-mancha.txt"

    # Parámetros del modelo
    seq_length = 50
    hidden_size = 256
    num_layers = 2
    batch_size = 128
    epochs = 20
    learning_rate = 0.001

    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(
        f"{Fore.CYAN}GENERACIÓN DE TEXTO CON REDES NEURONALES RECURRENTES (LSTM){Style.RESET_ALL}"
    )
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")

    # Procesar el texto
    processor = TextProcessor(file_path)
    stats = processor.get_stats()

    # Mostrar estadísticas
    print(f"\n{Fore.CYAN}--- ESTADÍSTICAS DEL TEXTO ---{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Total de caracteres: {Fore.WHITE}{stats['total_chars']:,}")
    print(f"{Fore.YELLOW}Tamaño del vocabulario: {Fore.WHITE}{stats['vocab_size']}")
    print(f"{Fore.YELLOW}Total de palabras: {Fore.WHITE}{stats['total_words']:,}")
    print(f"{Fore.YELLOW}Palabras únicas: {Fore.WHITE}{stats['unique_words']:,}")
    print(
        f"{Fore.YELLOW}Longitud media de palabra: {Fore.WHITE}{stats['avg_word_length']:.2f} caracteres"
    )

    print(f"\n{Fore.CYAN}Palabras más comunes:{Style.RESET_ALL}")
    for word, count in stats["most_common_words"]:
        print(f"  {Fore.GREEN}{word}: {Fore.WHITE}{count:,}")

    print(f"\n{Fore.CYAN}Caracteres más comunes:{Style.RESET_ALL}")
    for char, count in stats["most_common_chars"]:
        if char.isspace():
            char_display = "[espacio]"
        elif char == "\n":
            char_display = "[salto de línea]"
        else:
            char_display = char
        print(f"  {Fore.GREEN}{char_display}: {Fore.WHITE}{count:,}")

    print(f"\n{Fore.CYAN}Bigramas más comunes:{Style.RESET_ALL}")
    for bigram, count in stats["most_common_bigrams"]:
        # Reemplazar espacios y saltos de línea para visualización
        bigram_display = bigram.replace(" ", "␣").replace("\n", "⏎")
        print(f"  {Fore.GREEN}{bigram_display}: {Fore.WHITE}{count:,}")

    # Visualizar distribución de longitud de palabras
    plt.figure(figsize=(12, 6))
    sns.histplot(stats["word_lengths"], bins=30, kde=True, color="skyblue")
    plt.title("Distribución de longitud de palabras")
    plt.xlabel("Longitud de palabra")
    plt.ylabel("Frecuencia")
    plt.grid(True, alpha=0.3)
    plt.savefig("longitud_palabras.png")
    plt.show()

    # Visualizar palabras más comunes
    plt.figure(figsize=(12, 6))
    words, counts = zip(*stats["most_common_words"])
    plt.bar(words, counts, color="skyblue")
    plt.title("Palabras más comunes")
    plt.xlabel("Palabra")
    plt.ylabel("Frecuencia")
    plt.xticks(rotation=45)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("palabras_comunes.png")
    plt.show()

    # Visualizar caracteres más comunes
    plt.figure(figsize=(12, 6))
    # Transformar caracteres especiales para visualización
    char_labels = []
    char_counts = []
    for char, count in stats["most_common_chars"]:
        if char == " ":
            char_labels.append("[espacio]")
        elif char == "\n":
            char_labels.append("[salto]")
        else:
            char_labels.append(char)
        char_counts.append(count)

    plt.bar(char_labels, char_counts, color="lightgreen")
    plt.title("Caracteres más comunes")
    plt.xlabel("Carácter")
    plt.ylabel("Frecuencia")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("caracteres_comunes.png")
    plt.show()

    # Crear secuencias de entrenamiento
    print(f"\n{Fore.CYAN}PREPARACIÓN DE DATOS{Style.RESET_ALL}")
    X, y, sequences, next_chars = processor.create_sequences(seq_length)

    # Crear conjunto de datos y dataloader
    dataset = TextDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"{Fore.GREEN}Datos preparados y cargados en DataLoader.{Style.RESET_ALL}")

    # Crear y entrenar el modelo
    print(f"\n{Fore.CYAN}CREACIÓN Y ENTRENAMIENTO DEL MODELO{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Arquitectura del modelo:{Style.RESET_ALL}")
    print(f"  - Tamaño de entrada: {processor.vocab_size}")
    print(f"  - Tamaño de capa oculta: {hidden_size}")
    print(f"  - Número de capas LSTM: {num_layers}")
    print(f"  - Tamaño de salida: {processor.vocab_size}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Épocas: {epochs}")
    print(f"  - Tasa de aprendizaje: {learning_rate}")

    model = TextGenerationModel(
        input_size=processor.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        output_size=processor.vocab_size,
    ).to(device)

    # Mostrar resumen del modelo
    print(f"\n{Fore.YELLOW}Resumen del modelo:{Style.RESET_ALL}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  - Parámetros totales: {total_params:,}")
    print(f"  - Parámetros entrenables: {trainable_params:,}")

    # Visualizar la arquitectura del modelo como diagrama
    print(f"\n{Fore.YELLOW}Arquitectura del modelo:{Style.RESET_ALL}")
    print(
        f"  Input Layer ({processor.vocab_size}) → LSTM Layer 1 ({hidden_size}) → LSTM Layer 2 ({hidden_size}) → Dropout ({0.2}) → Fully Connected ({processor.vocab_size}) → Softmax"
    )

    print(f"\n{Fore.CYAN}INICIANDO ENTRENAMIENTO{Style.RESET_ALL}")
    seed_text = "en un lugar de la mancha"
    history = train_model(
        model, dataloader, epochs, learning_rate, processor, seed_text
    )

    # Visualizar métricas de entrenamiento
    plot_training_metrics(history)

    # Guardar el modelo
    model_path = "modelo_quijote.pth"
    torch.save(model.state_dict(), model_path)
    print(f"\n{Fore.GREEN}Modelo guardado como '{model_path}'{Style.RESET_ALL}")

    # Visualizar estados internos de la LSTM
    visualize_lstm_states(model, processor, seed_text, seq_length)

    # Generar texto con diferentes temperaturas
    print(f"\n{Fore.CYAN}GENERACIÓN DE TEXTO{Style.RESET_ALL}")

    temperatures = [0.2, 0.5, 0.7, 1.0, 1.5]
    generated_texts = {}
    probability_histories = {}

    for temp in temperatures:
        print(
            f"\n{Fore.YELLOW}Generando texto con temperatura {temp}:{Style.RESET_ALL}"
        )
        generated_text, prob_history = model.generate_text(
            processor, seed_text, length=500, temperature=temp
        )
        generated_texts[temp] = generated_text
        probability_histories[temp] = prob_history

        # Mostrar un fragmento del texto generado
        print(
            f"\n{Fore.GREEN}Fragmento del texto generado (primeros 200 caracteres):{Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}{generated_text[:200]}...{Style.RESET_ALL}")

        # Guardar el texto generado en un archivo
        output_file = f"texto_generado_temp_{temp}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(generated_text)
        print(
            f"{Fore.GREEN}Texto completo guardado en '{output_file}'{Style.RESET_ALL}"
        )

    # Visualizar distribución de probabilidades
    visualize_prediction_distribution(model, processor, seed_text, temperatures)

    # Visualizar entropía durante la generación para una temperatura específica
    plot_generation_entropy(probability_histories[1.0], processor)

    # Comparar diversidad léxica entre textos generados
    print(f"\n{Fore.CYAN}ANÁLISIS DE TEXTOS GENERADOS{Style.RESET_ALL}")

    diversity_scores = {}
    for temp, text in generated_texts.items():
        words = text.split()
        unique_words = len(set(words))
        total_words = len(words)
        diversity = unique_words / total_words if total_words > 0 else 0
        diversity_scores[temp] = diversity

        print(f"{Fore.YELLOW}Temperatura {temp}:{Style.RESET_ALL}")
        print(f"  - Total de palabras: {total_words}")
        print(f"  - Palabras únicas: {unique_words}")
        print(f"  - Diversidad léxica: {diversity:.4f}")

    # Visualizar diversidad léxica
    plt.figure(figsize=(10, 6))
    temps, scores = zip(*diversity_scores.items())
    plt.plot(
        temps,
        scores,
        marker="o",
        linestyle="-",
        color="purple",
        linewidth=2,
        markersize=8,
    )
    plt.title("Diversidad léxica vs. Temperatura")
    plt.xlabel("Temperatura")
    plt.ylabel("Diversidad léxica (palabras únicas / total)")
    plt.grid(True, alpha=0.3)

    # Añadir valores como etiquetas
    for temp, score in diversity_scores.items():
        plt.annotate(
            f"{score:.4f}",
            (temp, score),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.savefig("diversidad_lexica.png")
    plt.show()

    # Análisis de n-gramas para texto con temperatura 1.0
    jaccard_scores = {}
    for n in range(1, 4):  # Analizar 1-gramas, 2-gramas y 3-gramas
        jaccard_scores[n] = analyze_ngrams(
            processor.text[: len(generated_texts[1.0])], generated_texts[1.0], n=n
        )

    # Visualizar similitud de Jaccard para diferentes n-gramas
    plt.figure(figsize=(8, 5))
    plt.bar(jaccard_scores.keys(), jaccard_scores.values(), color="teal")
    plt.title("Similitud de Jaccard para diferentes n-gramas")
    plt.xlabel("Tamaño de n-grama")
    plt.ylabel("Similitud de Jaccard")
    plt.xticks(list(jaccard_scores.keys()))
    plt.grid(True, axis="y", alpha=0.3)

    # Añadir valores como etiquetas
    for n, score in jaccard_scores.items():
        plt.annotate(
            f"{score:.4f}",
            (n, score),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    plt.savefig("similitud_jaccard.png")
    plt.show()

    # Evaluación de calidad del texto generado
    quality_metrics = evaluate_text_quality(
        processor.text[: len(generated_texts[1.0])], generated_texts[1.0]
    )

    # Comparar textos generados con diferentes temperaturas
    print(
        f"\n{Fore.CYAN}COMPARACIÓN DE TEXTOS GENERADOS CON DIFERENTES TEMPERATURAS{Style.RESET_ALL}"
    )

    # Crear tabla comparativa
    comparison_data = []
    for temp in temperatures:
        text = generated_texts[temp]
        words = text.split()
        unique_words = len(set(words))
        total_words = len(words)
        diversity = unique_words / total_words if total_words > 0 else 0

        # Calcular entropía promedio
        avg_entropy = np.mean([p["entropy"] for p in probability_histories[temp]])

        # Calcular repetitividad (porcentaje de n-gramas repetidos)
        trigrams = ["".join(text[i : i + 3]) for i in range(len(text) - 2)]
        trigram_counter = Counter(trigrams)
        repeated_trigrams = sum(count > 1 for count in trigram_counter.values())
        repetitiveness = (
            repeated_trigrams / len(trigram_counter) if trigram_counter else 0
        )

        comparison_data.append(
            {
                "Temperatura": temp,
                "Diversidad léxica": diversity,
                "Entropía promedio": avg_entropy,
                "Repetitividad": repetitiveness,
                "Palabras totales": total_words,
                "Palabras únicas": unique_words,
            }
        )

    # Convertir a DataFrame para mejor visualización
    comparison_df = pd.DataFrame(comparison_data)
    print(comparison_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Visualizar comparación de métricas
    metrics = ["Diversidad léxica", "Entropía promedio", "Repetitividad"]

    fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4 * len(metrics)))

    for i, metric in enumerate(metrics):
        axes[i].plot(
            comparison_df["Temperatura"],
            comparison_df[metric],
            marker="o",
            linestyle="-",
            linewidth=2,
            markersize=8,
        )
        axes[i].set_title(f"{metric} vs. Temperatura")
        axes[i].set_xlabel("Temperatura")
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)

        # Añadir valores como etiquetas
        for j, val in enumerate(comparison_df[metric]):
            axes[i].annotate(
                f"{val:.4f}",
                (comparison_df["Temperatura"][j], val),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

    plt.tight_layout()
    plt.savefig("comparacion_metricas.png")
    plt.show()

    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}PROCESO COMPLETADO{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")


# Ejecutar la función principal
if __name__ == "__main__":
    start_time = time.time()
    main()
    total_time = time.time() - start_time
    minutes, seconds = divmod(total_time, 60)
    print(
        f"\n{Fore.GREEN}Tiempo total de ejecución: {int(minutes)} minutos y {seconds:.2f} segundos{Style.RESET_ALL}"
    )
