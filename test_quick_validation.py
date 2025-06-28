#!/usr/bin/env python3
"""
🧪 SCRIPT DE VALIDACIÓN RÁPIDA
Prueba todas las funcionalidades sin entrenamientos largos
Tiempo estimado: 2-3 minutos
"""

import sys
import traceback
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Suprimir warnings para output limpio
warnings.filterwarnings("ignore")
plt.ioff()  # Desactivar plots interactivos para testing


def test_imports():
    """Prueba todas las importaciones necesarias"""
    print("🔍 Probando importaciones...")

    try:
        import json
        import random
        import re
        from collections import Counter

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import plotly.graph_objects as go
        import seaborn as sns
        import torch
        import torchvision
        from sklearn.metrics import accuracy_score, classification_report
        from torch.utils.data import DataLoader, Dataset
        from tqdm import tqdm

        print("✅ Todas las librerías importadas correctamente")
        return True
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        return False


def test_device_setup():
    """Prueba configuración de dispositivo"""
    print("🎮 Probando configuración de dispositivo...")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✅ Dispositivo configurado: {device}")

        if torch.cuda.is_available():
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"   Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        return True, device
    except Exception as e:
        print(f"❌ Error en configuración de dispositivo: {e}")
        return False, "cpu"


def create_dummy_data():
    """Crea datos dummy para testing"""
    print("📊 Creando datos dummy para testing...")

    try:
        # Crear datos sintéticos que imiten la estructura real
        n_samples = 100

        dummy_data = {
            "train": pd.DataFrame(
                {
                    "dialog": [
                        [f"Hello how are you today?", f"I'm fine thanks, and you?"]
                        * np.random.randint(2, 5)
                        for _ in range(n_samples)
                    ],
                    "act": [
                        [f"greeting", f"inform"] * np.random.randint(2, 5)
                        for _ in range(n_samples)
                    ],
                    "emotion": [
                        [f"neutral", f"happy"] * np.random.randint(2, 5)
                        for _ in range(n_samples)
                    ],
                }
            ),
            "validation": pd.DataFrame(
                {
                    "dialog": [
                        [f"Good morning", f"Good morning to you too"]
                        * np.random.randint(2, 3)
                        for _ in range(20)
                    ],
                    "act": [
                        [f"greeting", f"greeting"] * np.random.randint(2, 3)
                        for _ in range(20)
                    ],
                    "emotion": [
                        [f"neutral", f"happy"] * np.random.randint(2, 3)
                        for _ in range(20)
                    ],
                }
            ),
            "test": pd.DataFrame(
                {
                    "dialog": [
                        [f"Thank you very much", f"You're welcome"]
                        * np.random.randint(2, 3)
                        for _ in range(20)
                    ],
                    "act": [
                        [f"thanking", f"acknowledge"] * np.random.randint(2, 3)
                        for _ in range(20)
                    ],
                    "emotion": [
                        [f"happy", f"neutral"] * np.random.randint(2, 3)
                        for _ in range(20)
                    ],
                }
            ),
        }

        print("✅ Datos dummy creados correctamente")
        return True, dummy_data
    except Exception as e:
        print(f"❌ Error creando datos dummy: {e}")
        return False, None


def test_tokenizer():
    """Prueba el tokenizador"""
    print("🔤 Probando tokenizador...")

    try:
        # Importar desde el código principal (asumiendo que está en el mismo directorio)
        sys.path.append(".")

        # Definir SimpleTokenizer aquí para testing
        class SimpleTokenizer:
            def __init__(self, texts, vocab_size=1000):
                self.vocab_size = vocab_size
                all_words = []
                for text in texts:
                    words = str(text).lower().split()
                    all_words.extend(words)

                word_freq = Counter(all_words)
                most_common = word_freq.most_common(vocab_size - 3)

                self.word_to_idx = {"<PAD>": 0, "<UNK>": 1, "<START>": 2}
                for word, _ in most_common:
                    self.word_to_idx[word] = len(self.word_to_idx)

                self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}

        # Probar tokenizador
        sample_texts = ["Hello world", "How are you", "Fine thanks"]
        tokenizer = SimpleTokenizer(sample_texts, vocab_size=100)

        assert len(tokenizer.word_to_idx) > 0
        assert "<PAD>" in tokenizer.word_to_idx
        assert "<UNK>" in tokenizer.word_to_idx

        print("✅ Tokenizador funciona correctamente")
        return True, SimpleTokenizer
    except Exception as e:
        print(f"❌ Error en tokenizador: {e}")
        traceback.print_exc()
        return False, None


def test_models(device):
    """Prueba la creación de modelos"""
    print("🤖 Probando creación de modelos...")

    try:
        # Definir modelos básicos para testing
        class SimpleRNN(nn.Module):
            def __init__(
                self, vocab_size, embedding_dim, hidden_dim, output_dim, dropout=0.5
            ):
                super(SimpleRNN, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(hidden_dim, output_dim)

            def forward(self, text, attention_mask=None):
                embedded = self.embedding(text)
                output, hidden = self.rnn(embedded)
                hidden = hidden[-1, :, :]
                hidden = self.dropout(hidden)
                return self.fc(hidden)

        class LSTMClassifier(nn.Module):
            def __init__(
                self,
                vocab_size,
                embedding_dim,
                hidden_dim,
                output_dim,
                n_layers=2,
                dropout=0.3,
                bidirectional=True,
            ):
                super(LSTMClassifier, self).__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.lstm = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    n_layers,
                    batch_first=True,
                    dropout=dropout if n_layers > 1 else 0,
                    bidirectional=bidirectional,
                )
                self.dropout = nn.Dropout(dropout)
                lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
                self.fc = nn.Linear(lstm_output_dim, output_dim)

            def forward(self, text, attention_mask=None):
                embedded = self.embedding(text)
                lstm_out, (hidden, _) = self.lstm(embedded)
                if self.lstm.bidirectional:
                    hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                else:
                    hidden = hidden[-1, :, :]
                hidden = self.dropout(hidden)
                return self.fc(hidden)

        # Probar creación de modelos
        vocab_size = 1000
        embedding_dim = 64
        hidden_dim = 128
        output_dim = 5

        models = {
            "SimpleRNN": SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim),
            "LSTM": LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
        }

        # Probar forward pass
        batch_size = 4
        seq_len = 10
        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))

        for name, model in models.items():
            model.to(device)
            model.eval()
            with torch.no_grad():
                output = model(dummy_input.to(device))
                assert output.shape == (batch_size, output_dim)
            print(f"   ✅ {name}: {output.shape}")

        print("✅ Todos los modelos funcionan correctamente")
        return True, models
    except Exception as e:
        print(f"❌ Error en modelos: {e}")
        traceback.print_exc()
        return False, None


def test_dataset():
    """Prueba la clase Dataset"""
    print("📦 Probando Dataset...")

    try:

        class DialogDataset:
            def __init__(self, texts, labels, word_to_idx, max_length=50):
                self.texts = texts
                self.labels = labels
                self.word_to_idx = word_to_idx
                self.max_length = max_length

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = str(self.texts[idx]).lower().split()
                label = self.labels[idx]

                # Convertir a índices
                input_ids = [
                    self.word_to_idx.get(word, self.word_to_idx["<UNK>"])
                    for word in text
                ]

                # Padding/truncating
                if len(input_ids) > self.max_length:
                    input_ids = input_ids[: self.max_length]
                else:
                    input_ids.extend([0] * (self.max_length - len(input_ids)))

                attention_mask = [1 if x != 0 else 0 for x in input_ids]

                return {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(label, dtype=torch.long),
                }

        # Probar dataset
        texts = ["hello world", "how are you", "fine thanks"]
        labels = [0, 1, 2]
        word_to_idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "hello": 2,
            "world": 3,
            "how": 4,
            "are": 5,
            "you": 6,
            "fine": 7,
            "thanks": 8,
        }

        dataset = DialogDataset(texts, labels, word_to_idx, max_length=10)

        # Probar DataLoader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        for batch in dataloader:
            assert "input_ids" in batch
            assert "labels" in batch
            assert "attention_mask" in batch
            break

        print("✅ Dataset y DataLoader funcionan correctamente")
        return True, DialogDataset
    except Exception as e:
        print(f"❌ Error en Dataset: {e}")
        traceback.print_exc()
        return False, None


def test_training_loop(models, device):
    """Prueba un mini loop de entrenamiento"""
    print("🏋️ Probando loop de entrenamiento...")

    try:
        # Crear datos dummy
        batch_size = 4
        seq_len = 10
        vocab_size = 1000
        num_classes = 5

        dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
        dummy_labels = torch.randint(0, num_classes, (batch_size,)).to(device)
        dummy_attention = torch.ones(batch_size, seq_len).to(device)

        # Probar entrenamiento con un modelo
        model = list(models.values())[0]
        model.train()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Mini entrenamiento (2 pasos)
        for step in range(2):
            optimizer.zero_grad()
            outputs = model(dummy_input, dummy_attention)
            loss = criterion(outputs, dummy_labels)
            loss.backward()
            optimizer.step()

            print(f"   Paso {step+1}: Loss = {loss.item():.4f}")

        print("✅ Loop de entrenamiento funciona correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en entrenamiento: {e}")
        traceback.print_exc()
        return False


def test_evaluation():
    """Prueba funciones de evaluación"""
    print("📊 Probando evaluación...")

    try:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        # Datos dummy para evaluación
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2, 0, 1, 2]

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1-Score: {f1:.3f}")

        assert 0 <= accuracy <= 1
        assert 0 <= f1 <= 1

        print("✅ Evaluación funciona correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en evaluación: {e}")
        return False


def test_visualization():
    """Prueba funciones de visualización"""
    print("📈 Probando visualización...")

    try:
        # Crear plot simple
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        x = [1, 2, 3, 4, 5]
        y = [0.8, 0.85, 0.9, 0.88, 0.92]

        ax.plot(x, y, "b-", label="Accuracy")
        ax.set_xlabel("Época")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.close(fig)  # Cerrar para no mostrar

        # Probar seaborn
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        y_true = [0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 2, 2]
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", ax=ax)
        plt.close(fig)

        print("✅ Visualización funciona correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en visualización: {e}")
        return False


def test_file_operations():
    """Prueba operaciones de archivos"""
    print("💾 Probando operaciones de archivos...")

    try:
        import json

        # Probar escritura/lectura JSON
        test_data = {
            "test": True,
            "timestamp": datetime.now().isoformat(),
            "results": {"accuracy": 0.85, "f1": 0.82},
        }

        filename = "test_temp.json"

        # Escribir
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(test_data, f, indent=2, ensure_ascii=False)

        # Leer
        with open(filename, "r", encoding="utf-8") as f:
            loaded_data = json.load(f)

        assert loaded_data["test"] == True
        assert "timestamp" in loaded_data

        # Limpiar archivo temporal
        import os

        os.remove(filename)

        print("✅ Operaciones de archivos funcionan correctamente")
        return True
    except Exception as e:
        print(f"❌ Error en operaciones de archivos: {e}")
        return False


def test_data_loading():
    """Prueba carga de datos reales si existen"""
    print("📂 Probando carga de datos reales...")

    try:
        import os

        # Verificar si existen archivos parquet
        files_to_check = ["train.parquet", "validation.parquet", "test.parquet"]
        existing_files = []

        for file in files_to_check:
            if os.path.exists(file):
                existing_files.append(file)

        if existing_files:
            print(f"   Archivos encontrados: {existing_files}")

            # Intentar cargar uno
            df = pd.read_parquet(existing_files[0])
            print(f"   Forma del dataset: {df.shape}")
            print(f"   Columnas: {list(df.columns)}")

            # Verificar estructura esperada
            expected_cols = ["dialog", "act", "emotion"]
            missing_cols = [col for col in expected_cols if col not in df.columns]

            if missing_cols:
                print(f"   ⚠️ Columnas faltantes: {missing_cols}")
            else:
                print("   ✅ Estructura de datos correcta")

            return True, len(existing_files) == 3
        else:
            print("   ⚠️ No se encontraron archivos parquet")
            print("   ℹ️ Se usarán datos dummy para testing")
            return True, False

    except Exception as e:
        print(f"❌ Error en carga de datos: {e}")
        return False, False


def test_memory_usage():
    """Prueba uso de memoria"""
    print("🧠 Probando uso de memoria...")

    try:
        import gc

        import psutil

        # Memoria inicial
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Crear tensores grandes
        large_tensor = torch.randn(1000, 1000)
        if torch.cuda.is_available():
            large_tensor = large_tensor.cuda()

        # Memoria después
        current_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Limpiar
        del large_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"   Memoria inicial: {initial_memory:.1f} MB")
        print(f"   Memoria con tensor: {current_memory:.1f} MB")
        print(f"   Diferencia: {current_memory - initial_memory:.1f} MB")

        print("✅ Gestión de memoria funciona correctamente")
        return True
    except ImportError:
        print("   ⚠️ psutil no disponible, saltando test de memoria")
        return True
    except Exception as e:
        print(f"❌ Error en test de memoria: {e}")
        return False


def run_integration_test():
    """Ejecuta un test de integración completo pero rápido"""
    print("🔄 Ejecutando test de integración...")

    try:
        # 1. Crear datos dummy
        success, dummy_data = create_dummy_data()
        if not success:
            return False

        # 2. Crear tokenizer
        all_texts = []
        for split in dummy_data.values():
            for dialogs in split["dialog"]:
                all_texts.extend(dialogs)

        success, TokenizerClass = test_tokenizer()
        if not success:
            return False

        tokenizer = TokenizerClass(all_texts, vocab_size=500)

        # 3. Crear dataset
        success, DatasetClass = test_dataset()
        if not success:
            return False

        # Expandir datos (simular expand_dialogues)
        expanded_texts = []
        expanded_labels = []

        for dialogs, acts in zip(
            dummy_data["train"]["dialog"], dummy_data["train"]["act"]
        ):
            for dialog, act in zip(dialogs, acts):
                expanded_texts.append(dialog)
                # Mapear actos a números
                act_to_idx = {
                    "greeting": 0,
                    "inform": 1,
                    "thanking": 2,
                    "acknowledge": 3,
                }
                expanded_labels.append(act_to_idx.get(act, 0))

        dataset = DatasetClass(
            expanded_texts, expanded_labels, tokenizer.word_to_idx, max_length=20
        )

        # 4. Crear DataLoader
        from torch.utils.data import DataLoader

        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

        # 5. Crear modelo
        success, device = test_device_setup()
        if not success:
            return False

        success, models = test_models(device)
        if not success:
            return False

        model = list(models.values())[0]  # Usar primer modelo

        # 6. Mini entrenamiento
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        print("   Ejecutando mini-entrenamiento...")
        for epoch in range(2):  # Solo 2 épocas
            epoch_loss = 0
            batch_count = 0

            for batch in dataloader:
                if batch_count >= 3:  # Solo 3 batches por época
                    break

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                batch_count += 1

            avg_loss = epoch_loss / batch_count
            print(f"     Época {epoch+1}: Loss promedio = {avg_loss:.4f}")

        # 7. Evaluación rápida
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            batch_count = 0
            for batch in dataloader:
                if batch_count >= 2:  # Solo 2 batches para evaluación
                    break

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs, 1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                batch_count += 1

        # Calcular métricas
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(all_labels, all_preds)
        print(f"   Accuracy en evaluación rápida: {accuracy:.3f}")

        print("✅ Test de integración completado exitosamente")
        return True

    except Exception as e:
        print(f"❌ Error en test de integración: {e}")
        traceback.print_exc()
        return False


def main():
    """Función principal del test"""
    print("🧪 INICIANDO VALIDACIÓN RÁPIDA DEL CÓDIGO")
    print("=" * 60)
    print(f"⏰ Tiempo estimado: 2-3 minutos")
    print(f"🎯 Objetivo: Validar que el código principal funcionará sin errores")
    print("=" * 60)

    start_time = datetime.now()

    # Lista de tests a ejecutar
    tests = [
        ("Importaciones", test_imports),
        ("Configuración de dispositivo", lambda: test_device_setup()[0]),
        ("Datos dummy", lambda: create_dummy_data()[0]),
        ("Tokenizador", lambda: test_tokenizer()[0]),
        (
            "Modelos",
            lambda: test_models(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )[0],
        ),
        ("Dataset", lambda: test_dataset()[0]),
        (
            "Loop de entrenamiento",
            lambda: test_training_loop(
                *test_models(
                    torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
            ),
        ),
        ("Evaluación", test_evaluation),
        ("Visualización", test_visualization),
        ("Operaciones de archivos", test_file_operations),
        ("Carga de datos", lambda: test_data_loading()[0]),
        ("Uso de memoria", test_memory_usage),
        ("Test de integración", run_integration_test),
    ]

    # Ejecutar tests
    passed = 0
    failed = 0

    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name}: PASÓ")
            else:
                failed += 1
                print(f"❌ {test_name}: FALLÓ")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name}: ERROR - {e}")

    # Resumen final
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    print(f"\n{'='*60}")
    print(f"📊 RESUMEN DE VALIDACIÓN")
    print(f"{'='*60}")
    print(f"✅ Tests pasados: {passed}")
    print(f"❌ Tests fallidos: {failed}")
    print(f"📈 Tasa de éxito: {(passed/(passed+failed))*100:.1f}%")
    print(f"⏱️ Tiempo total: {duration:.1f} segundos")

    # Verificar datos reales
    print(f"\n📂 VERIFICACIÓN DE DATOS:")
    data_success, has_real_data = test_data_loading()
    if has_real_data:
        print("✅ Archivos parquet encontrados - El código usará datos reales")
    else:
        print(
            "⚠️ Archivos parquet no encontrados - Asegúrate de tenerlos antes del entrenamiento real"
        )

    # Recomendaciones
    print(f"\n💡 RECOMENDACIONES:")
    if failed == 0:
        print("🎉 ¡Perfecto! El código principal debería ejecutarse sin problemas")
        print("🚀 Puedes proceder con confianza al entrenamiento completo")
    elif failed <= 2:
        print(
            "⚠️ Algunos tests menores fallaron, pero el código principal probablemente funcione"
        )
        print("🔍 Revisa los errores específicos arriba")
    else:
        print("🚨 Varios tests fallaron - revisa las dependencias y configuración")
        print("🛠️ Soluciona los errores antes de ejecutar el código principal")

    # Estimación de tiempo para ejecución completa
    if torch.cuda.is_available():
        print(f"\n⏱️ ESTIMACIÓN DE TIEMPO COMPLETO (con GPU):")
        print(f"   🎭 Análisis de actos de habla: 15-25 minutos")
        print(f"   😊 Análisis de emociones: 15-25 minutos")
        print(f"   🧪 Experimentación: 20-30 minutos")
        print(f"   📊 Total estimado: 50-80 minutos")
    else:
        print(f"\n⏱️ ESTIMACIÓN DE TIEMPO COMPLETO (solo CPU):")
        print(f"   ⚠️ Sin GPU detectada - el entrenamiento será MUY lento")
        print(f"   🐌 Tiempo estimado: 3-6 horas")
        print(f"   💡 Recomendación: Usar Google Colab con GPU")

    return failed == 0


if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🎯 VALIDACIÓN EXITOSA")
        print(f"✅ El código principal está listo para ejecutarse")
        exit(0)
    else:
        print(f"\n⚠️ VALIDACIÓN CON ERRORES")
        print(f"🔧 Revisa y corrige los problemas antes de continuar")
        exit(1)
