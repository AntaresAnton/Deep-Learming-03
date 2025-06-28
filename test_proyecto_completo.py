#!/usr/bin/env python3
"""
🧪 VALIDADOR COMPLETO PARA proyecto_deep_learning_dialogos.py
Prueba todas las funciones del archivo principal sin ejecutar el entrenamiento completo
INCLUYE ESTIMACIÓN DE TIEMPO DE EJECUCIÓN
"""

import importlib.util
import os
import sys
import time
import traceback
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import torch


def load_main_module():
    """Carga el módulo principal"""
    try:
        spec = importlib.util.spec_from_file_location(
            "main_project", "proyecto_deep_learning_dialogos.py"
        )
        main_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(main_module)
        return main_module, True
    except Exception as e:
        print(f"❌ Error cargando módulo principal: {e}")
        return None, False


def estimate_execution_time():
    """Estima el tiempo de ejecución del proyecto completo"""
    print("⏱️ ESTIMANDO TIEMPO DE EJECUCIÓN DEL PROYECTO COMPLETO")
    print("-" * 60)

    try:
        # Verificar archivos de datos
        if not all(
            os.path.exists(f)
            for f in ["train.parquet", "validation.parquet", "test.parquet"]
        ):
            print("❌ No se pueden estimar tiempos sin archivos de datos")
            return None

        # Cargar datos para análisis
        print("📊 Analizando tamaños de datos...")
        train_df = pd.read_parquet("train.parquet")
        val_df = pd.read_parquet("validation.parquet")
        test_df = pd.read_parquet("test.parquet")

        # Calcular estadísticas
        total_train_dialogs = sum(
            len(row) if isinstance(row, list) else 1 for row in train_df["dialog"]
        )
        total_val_dialogs = sum(
            len(row) if isinstance(row, list) else 1 for row in val_df["dialog"]
        )
        total_test_dialogs = sum(
            len(row) if isinstance(row, list) else 1 for row in test_df["dialog"]
        )

        total_expanded = total_train_dialogs + total_val_dialogs + total_test_dialogs

        print(f"   📋 Datos originales:")
        print(f"      Train: {len(train_df):,} filas")
        print(f"      Validation: {len(val_df):,} filas")
        print(f"      Test: {len(test_df):,} filas")
        print(f"   📈 Datos expandidos estimados:")
        print(f"      Train: {total_train_dialogs:,} diálogos")
        print(f"      Validation: {total_val_dialogs:,} diálogos")
        print(f"      Test: {total_test_dialogs:,} diálogos")
        print(f"      Total: {total_expanded:,} diálogos")

        # Configuración del proyecto
        n_epochs = 15
        n_models = 4  # SimpleRNN, LSTM, GRU, Transformer
        n_tasks = 2  # Actos de habla y emociones

        # Detectar dispositivo
        device = "GPU" if torch.cuda.is_available() else "CPU"
        if device == "GPU":
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🎮 Dispositivo: {device} ({gpu_name})")
        else:
            print(f"🎮 Dispositivo: {device}")

        # Estimaciones de tiempo basadas en el dispositivo y tamaño de datos
        if device == "GPU":
            # Tiempos con GPU (más rápidos)
            base_time_per_epoch = 30  # segundos base por época
            data_loading_time = 60  # 1 minuto
            preprocessing_time = 120  # 2 minutos
            evaluation_time = 30  # 30 segundos por modelo por tarea
            visualization_time = 60  # 1 minuto

            # Ajustar por tamaño de datos
            if total_expanded > 50000:
                base_time_per_epoch *= 2
                preprocessing_time *= 1.5
            elif total_expanded > 100000:
                base_time_per_epoch *= 3
                preprocessing_time *= 2

        else:
            # Tiempos con CPU (más lentos)
            base_time_per_epoch = 120  # 2 minutos base por época
            data_loading_time = 120  # 2 minutos
            preprocessing_time = 300  # 5 minutos
            evaluation_time = 90  # 1.5 minutos por modelo por tarea
            visualization_time = 120  # 2 minutos

            # Ajustar por tamaño de datos
            if total_expanded > 50000:
                base_time_per_epoch *= 2.5
                preprocessing_time *= 2
            elif total_expanded > 100000:
                base_time_per_epoch *= 4
                preprocessing_time *= 3

        # Cálculos detallados
        training_time_per_task = n_epochs * base_time_per_epoch * n_models
        total_training_time = training_time_per_task * n_tasks
        total_evaluation_time = evaluation_time * n_models * n_tasks

        # Tiempo total
        total_time = (
            data_loading_time
            + preprocessing_time
            + total_training_time
            + total_evaluation_time
            + visualization_time
        )

        print(f"\n⏱️ ESTIMACIÓN DETALLADA DE TIEMPOS:")
        print(f"   📂 Carga de datos: {timedelta(seconds=data_loading_time)}")
        print(f"   🔄 Preprocesamiento: {timedelta(seconds=preprocessing_time)}")
        print(f"   🏋️ Entrenamiento:")
        print(f"      Por época por modelo: {base_time_per_epoch}s")
        print(f"      Por tarea: {timedelta(seconds=training_time_per_task)}")
        print(f"      Total: {timedelta(seconds=total_training_time)}")
        print(f"   📊 Evaluación: {timedelta(seconds=total_evaluation_time)}")
        print(f"   📈 Visualizaciones: {timedelta(seconds=visualization_time)}")

        print(f"\n🎯 TIEMPO TOTAL ESTIMADO: {timedelta(seconds=total_time)}")
        print(f"   ({total_time/3600:.1f} horas)")

        # Desglose por componente
        print(f"\n📋 DESGLOSE POR COMPONENTE:")
        components = [
            ("Carga de datos", data_loading_time),
            ("Preprocesamiento", preprocessing_time),
            ("Entrenamiento", total_training_time),
            ("Evaluación", total_evaluation_time),
            ("Visualizaciones", visualization_time),
        ]

        for name, time_sec in components:
            percentage = (time_sec / total_time) * 100
            print(f"   {name:<15}: {timedelta(seconds=time_sec)} ({percentage:.1f}%)")

        # Recomendaciones basadas en el tiempo estimado
        print(f"\n💡 RECOMENDACIONES:")

        if total_time > 7200:  # Más de 2 horas
            print(f"   ⚠️ TIEMPO LARGO (>2 horas)")
            print(f"   🔧 Considera estas opciones:")
            print(f"      • Reducir épocas de 15 a 10")
            print(f"      • Usar solo 2-3 modelos principales")
            print(f"      • Ejecutar en horario nocturno")
            print(f"      • Usar GPU si está disponible")
        elif total_time > 3600:  # Más de 1 hora
            print(f"   ⏰ TIEMPO MODERADO (1-2 horas)")
            print(f"   👍 Tiempo razonable para ejecución completa")
            print(f"   💡 Puedes ejecutar durante una pausa")
        else:
            print(f"   ✅ TIEMPO CORTO (<1 hora)")
            print(f"   🚀 Perfecto para ejecución inmediata")

        # Opciones de optimización
        print(f"\n⚡ OPCIONES DE OPTIMIZACIÓN:")

        # Versión rápida
        quick_time = total_time * 0.3  # 30% del tiempo original
        print(
            f"   🏃 Versión rápida (5 épocas, 2 modelos): ~{timedelta(seconds=quick_time)}"
        )

        # Solo validación
        validation_time = (
            preprocessing_time
            + (base_time_per_epoch * 2 * n_tasks)
            + (evaluation_time * 2 * n_tasks)
        )
        print(
            f"   🧪 Solo validación (2 épocas): ~{timedelta(seconds=validation_time)}"
        )

        # Por partes
        per_task_time = total_time / n_tasks
        print(f"   📊 Por tarea individual: ~{timedelta(seconds=per_task_time)}")

        return {
            "total_time": total_time,
            "components": dict(components),
            "device": device,
            "data_size": total_expanded,
            "recommendations": (
                "fast"
                if total_time < 3600
                else "moderate" if total_time < 7200 else "slow"
            ),
        }

    except Exception as e:
        print(f"❌ Error estimando tiempo: {e}")
        return None


def benchmark_training_speed(main_module):
    """Hace un benchmark rápido para calibrar estimaciones"""
    print("🏃 BENCHMARK DE VELOCIDAD DE ENTRENAMIENTO")
    print("-" * 50)

    try:
        # Crear datos dummy pequeños para benchmark
        texts = ["hello world"] * 100
        labels = [0, 1] * 50
        word_to_idx = {"<PAD>": 0, "<UNK>": 1, "hello": 2, "world": 3}

        dataset = main_module.DialogDataset(texts, labels, word_to_idx, max_length=10)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16)

        # Crear modelo simple
        model = main_module.SimpleRNN(10, 32, 64, 2)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Medir tiempo de una época
        start_time = time.time()

        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_time = time.time() - start_time

        print(f"   ⏱️ Tiempo por época (100 muestras): {epoch_time:.2f}s")
        print(f"   📊 Velocidad estimada: {100/epoch_time:.1f} muestras/segundo")

        # Extrapolar para dataset real
        if os.path.exists("train.parquet"):
            train_df = pd.read_parquet("train.parquet")
            estimated_samples = sum(
                len(row) if isinstance(row, list) else 1 for row in train_df["dialog"]
            )
            estimated_time_per_epoch = (estimated_samples / 100) * epoch_time

            print(
                f"   🎯 Tiempo estimado por época (dataset real): {estimated_time_per_epoch:.1f}s"
            )
            print(
                f"   🏋️ Tiempo estimado para 15 épocas: {timedelta(seconds=estimated_time_per_epoch * 15)}"
            )

        return epoch_time

    except Exception as e:
        print(f"   ❌ Error en benchmark: {e}")
        return None


def test_tokenizer_class(main_module):
    """Prueba la clase SimpleTokenizer"""
    print("🔤 Probando SimpleTokenizer...")
    try:
        texts = ["hello world", "this is a test", "tokenizer works great"]
        tokenizer = main_module.SimpleTokenizer(texts, vocab_size=100)

        assert hasattr(tokenizer, "word_to_idx")
        assert hasattr(tokenizer, "idx_to_word")
        assert "<PAD>" in tokenizer.word_to_idx
        assert "<UNK>" in tokenizer.word_to_idx

        print("   ✅ SimpleTokenizer funciona correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error en SimpleTokenizer: {e}")
        return False


def test_dataset_class(main_module):
    """Prueba la clase DialogDataset"""
    print("📊 Probando DialogDataset...")
    try:
        texts = ["hello world", "this is a test"]
        labels = [0, 1]
        word_to_idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "hello": 2,
            "world": 3,
            "this": 4,
            "is": 5,
            "a": 6,
            "test": 7,
        }

        dataset = main_module.DialogDataset(texts, labels, word_to_idx, max_length=10)

        assert len(dataset) == 2
        sample = dataset[0]
        assert "input_ids" in sample
        assert "attention_mask" in sample
        assert "labels" in sample

        print("   ✅ DialogDataset funciona correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error en DialogDataset: {e}")
        return False


def test_model_classes(main_module):
    """Prueba todas las clases de modelos"""
    print("🧠 Probando clases de modelos...")

    vocab_size = 1000
    embedding_dim = 64
    hidden_dim = 128
    output_dim = 5

    models_to_test = [
        ("SimpleRNN", main_module.SimpleRNN),
        ("LSTMClassifier", main_module.LSTMClassifier),
        ("GRUClassifier", main_module.GRUClassifier),
        ("TransformerClassifier", main_module.TransformerClassifier),
    ]

    all_passed = True

    for model_name, model_class in models_to_test:
        try:
            print(f"   🔍 Probando {model_name}...")

            if model_name == "TransformerClassifier":
                model = model_class(vocab_size, embedding_dim, output_dim)
            else:
                model = model_class(vocab_size, embedding_dim, hidden_dim, output_dim)

            # Probar forward pass
            batch_size = 4
            seq_length = 20
            input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
            attention_mask = torch.ones(batch_size, seq_length)

            with torch.no_grad():
                output = model(input_ids, attention_mask)

            assert output.shape == (batch_size, output_dim)
            print(f"      ✅ {model_name} funciona correctamente")

        except Exception as e:
            print(f"      ❌ Error en {model_name}: {e}")
            all_passed = False

    return all_passed


def test_utility_functions(main_module):
    """Prueba funciones utilitarias"""
    print("🛠️ Probando funciones utilitarias...")

    functions_to_test = [
        "expand_dialogues",
        "plot_training_history",
        "plot_confusion_matrix",
        "plot_model_comparison",
        "visualize_model_architecture",
        "save_results",
        "analyze_model_performance",
        "hyperparameter_analysis",
    ]

    all_passed = True

    for func_name in functions_to_test:
        try:
            if hasattr(main_module, func_name):
                print(f"   ✅ {func_name} encontrada")
            else:
                print(f"   ❌ {func_name} no encontrada")
                all_passed = False
        except Exception as e:
            print(f"   ❌ Error verificando {func_name}: {e}")
            all_passed = False

    return all_passed


def test_expand_dialogues_function(main_module):
    """Prueba específica de la función expand_dialogues"""
    print("🔄 Probando función expand_dialogues...")
    try:
        # Crear datos de prueba
        test_data = pd.DataFrame(
            {
                "dialog": [
                    ["Hello", "How are you?", "I am fine"],
                    ["Good morning", "Nice weather today"],
                ],
                "act": [["greeting", "question", "inform"], ["greeting", "inform"]],
                "emotion": [["neutral", "curious", "happy"], ["happy", "happy"]],
            }
        )

        expanded = main_module.expand_dialogues(test_data)

        assert len(expanded) == 5  # 3 + 2 diálogos expandidos
        assert "dialog" in expanded.columns
        assert "act" in expanded.columns
        assert "emotion" in expanded.columns

        print("   ✅ expand_dialogues funciona correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error en expand_dialogues: {e}")
        return False


def test_training_components(main_module):
    """Prueba componentes de entrenamiento sin entrenar realmente"""
    print("🏋️ Probando componentes de entrenamiento...")
    try:
        # Crear modelo y datos dummy
        model = main_module.SimpleRNN(100, 32, 64, 3)

        # Crear datos dummy
        texts = ["hello world", "test sentence", "another example"]
        labels = [0, 1, 2]
        word_to_idx = {
            "<PAD>": 0,
            "<UNK>": 1,
            "hello": 2,
            "world": 3,
            "test": 4,
            "sentence": 5,
            "another": 6,
            "example": 7,
        }

        dataset = main_module.DialogDataset(texts, labels, word_to_idx, max_length=10)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Probar un paso de entrenamiento
        optimizer = torch.optim.Adam(model.parameters())
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        for batch in dataloader:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            break  # Solo un batch

        print("   ✅ Componentes de entrenamiento funcionan correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error en componentes de entrenamiento: {e}")
        return False


def test_evaluation_components(main_module):
    """Prueba componentes de evaluación"""
    print("📊 Probando componentes de evaluación...")
    try:
        # Simular resultados de evaluación
        dummy_results = {
            "accuracy": 0.85,
            "precision": 0.83,
            "recall": 0.82,
            "f1": 0.84,
            "classification_report": {
                "0": {"precision": 0.8, "recall": 0.9, "f1-score": 0.85}
            },
            "confusion_matrix": np.array([[10, 2], [1, 8]]),
            "predictions": [0, 1, 0, 1],
            "true_labels": [0, 1, 1, 1],
        }

        # Probar análisis de rendimiento
        results_dict = {"LSTM": dummy_results, "GRU": dummy_results}

        # Esto debería funcionar sin errores
        analysis_df = main_module.analyze_model_performance(results_dict, "Test Task")

        assert len(analysis_df) == 2
        assert "Modelo" in analysis_df.columns

        print("   ✅ Componentes de evaluación funcionan correctamente")
        return True
    except Exception as e:
        print(f"   ❌ Error en componentes de evaluación: {e}")
        return False


def test_data_file_requirements():
    """Verifica si los archivos de datos están presentes"""
    print("📂 Verificando archivos de datos requeridos...")

    required_files = ["train.parquet", "validation.parquet", "test.parquet"]
    found_files = []
    missing_files = []

    for file in required_files:
        if os.path.exists(file):
            found_files.append(file)
            size_mb = os.path.getsize(file) / (1024 * 1024)
            print(f"   ✅ {file} encontrado ({size_mb:.1f} MB)")
        else:
            missing_files.append(file)
            print(f"   ❌ {file} no encontrado")

    if missing_files:
        print(f"   ⚠️ Archivos faltantes: {missing_files}")
        print(f"   ℹ️ El proyecto usará datos dummy para testing")
        return False
    else:
        print(f"   ✅ Todos los archivos de datos están presentes")
        return True


def run_integration_test(main_module):
    """Ejecuta un test de integración completo pero rápido"""
    print("🔄 Ejecutando test de integración...")
    try:
        # Crear datos dummy que simulen la estructura real
        dummy_train = pd.DataFrame(
            {
                "dialog": [
                    ["Hello", "How are you?"],
                    ["Good morning", "Nice day"],
                    ["Thank you", "You are welcome"],
                ],
                "act": [
                    ["greeting", "question"],
                    ["greeting", "inform"],
                    ["thanking", "acknowledge"],
                ],
                "emotion": [
                    ["neutral", "curious"],
                    ["happy", "happy"],
                    ["grateful", "kind"],
                ],
            }
        )

        # Expandir datos
        expanded_train = main_module.expand_dialogues(dummy_train)

        # Crear tokenizer
        all_texts = list(expanded_train["dialog"])
        tokenizer = main_module.SimpleTokenizer(all_texts, vocab_size=50)

        # Crear mapeo de etiquetas
        unique_acts = sorted(list(set(expanded_train["act"])))
        act_to_idx = {act: idx for idx, act in enumerate(unique_acts)}

        # Convertir etiquetas
        train_labels = [act_to_idx[act] for act in expanded_train["act"]]

        # Crear dataset
        dataset = main_module.DialogDataset(
            expanded_train["dialog"], train_labels, tokenizer.word_to_idx, max_length=20
        )

        # Crear dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

        # Crear modelo simple
        model = main_module.SimpleRNN(
            vocab_size=tokenizer.vocab_size,
            embedding_dim=32,
            hidden_dim=64,
            output_dim=len(unique_acts),
        )

        # Mini entrenamiento (1 época, pocos batches)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        model.train()
        total_loss = 0
        batch_count = 0

        for batch in dataloader:
            if batch_count >= 2:  # Solo 2 batches
                break

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels = batch["labels"]

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(
            f"   ✅ Test de integración completado (pérdida promedio: {avg_loss:.4f})"
        )
        return True

    except Exception as e:
        print(f"   ❌ Error en test de integración: {e}")
        traceback.print_exc()
        return False


def generate_execution_plan(time_estimate):
    """Genera un plan de ejecución basado en la estimación de tiempo"""
    print("📋 PLAN DE EJECUCIÓN RECOMENDADO")
    print("-" * 50)

    if not time_estimate:
        print("❌ No se pudo generar plan sin estimación de tiempo")
        return

    total_hours = time_estimate["total_time"] / 3600

    if total_hours < 1:
        print("🚀 EJECUCIÓN INMEDIATA")
        print("   ✅ Tiempo estimado: < 1 hora")
        print("   💡 Recomendación: Ejecutar ahora")
        print("   📝 Comando: python proyecto_deep_learning_dialogos.py")

    elif total_hours < 3:
        print("⏰ EJECUCIÓN PROGRAMADA")
        print(f"   ⏱️ Tiempo estimado: {total_hours:.1f} horas")
        print("   💡 Recomendación: Ejecutar durante pausa larga")
        print("   📝 Opciones:")
        print("      • Ejecución completa: python proyecto_deep_learning_dialogos.py")
        print("      • Solo validación: python test_proyecto_completo.py")

    else:
        print("🌙 EJECUCIÓN NOCTURNA")
        print(f"   ⏱️ Tiempo estimado: {total_hours:.1f} horas")
        print("   💡 Recomendación: Ejecutar durante la noche")
        print("   📝 Opciones:")
        print(
            "      • Ejecución completa: nohup python proyecto_deep_learning_dialogos.py > output.log 2>&1 &"
        )
        print("      • Por partes: Ejecutar cada tarea por separado")

    print(f"\n⚡ ALTERNATIVAS RÁPIDAS:")
    print(f"   🧪 Solo validación: python test_proyecto_completo.py")
    print(f"   🎯 Demos específicos: python ejecutar_partes_especificas.py")
    print(f"   📊 Benchmark: python ejecutar_partes_especificas.py quick")


def main():
    """Función principal del validador"""
    print("🧪 VALIDADOR COMPLETO PARA proyecto_deep_learning_dialogos.py")
    print("INCLUYE ESTIMACIÓN DE TIEMPO DE EJECUCIÓN")
    print("=" * 70)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Cargar módulo principal
    print("\n📦 Cargando módulo principal...")
    main_module, module_loaded = load_main_module()

    if not module_loaded:
        print("❌ No se pudo cargar el módulo principal. Abortando tests.")
        return False

    print("✅ Módulo principal cargado exitosamente")

    # 2. Verificar archivos de datos
    data_files_present = test_data_file_requirements()

    # 3. Estimación de tiempo de ejecución
    print(f"\n{'='*70}")
    time_estimate = estimate_execution_time()
    print(f"{'='*70}")

    # 4. Benchmark de velocidad (opcional)
    if data_files_present:
        print(f"\n{'='*70}")
        benchmark_time = benchmark_training_speed(main_module)
        print(f"{'='*70}")

    # 5. Tests de componentes individuales
    test_results = []

    print(f"\n🔍 EJECUTANDO TESTS DE COMPONENTES:")
    print("-" * 50)

    # Test tokenizer
    result = test_tokenizer_class(main_module)
    test_results.append(("SimpleTokenizer", result))

    # Test dataset
    result = test_dataset_class(main_module)
    test_results.append(("DialogDataset", result))

    # Test modelos
    result = test_model_classes(main_module)
    test_results.append(("Clases de Modelos", result))

    # Test funciones utilitarias
    result = test_utility_functions(main_module)

    test_results.append(("Funciones Utilitarias", result))

    # Test expand_dialogues específico
    result = test_expand_dialogues_function(main_module)
    test_results.append(("expand_dialogues", result))

    # Test componentes de entrenamiento
    result = test_training_components(main_module)
    test_results.append(("Componentes de Entrenamiento", result))

    # Test componentes de evaluación
    result = test_evaluation_components(main_module)
    test_results.append(("Componentes de Evaluación", result))

    # 6. Test de integración
    print(f"\n🔄 EJECUTANDO TEST DE INTEGRACIÓN:")
    print("-" * 50)

    integration_result = run_integration_test(main_module)
    test_results.append(("Test de Integración", integration_result))

    # 7. Resumen de resultados
    print(f"\n📊 RESUMEN DE RESULTADOS:")
    print("=" * 70)

    passed_tests = 0
    total_tests = len(test_results)

    for test_name, result in test_results:
        status = "✅ PASÓ" if result else "❌ FALLÓ"
        print(f"{status:10} | {test_name}")
        if result:
            passed_tests += 1

    # 8. Estadísticas finales
    success_rate = (passed_tests / total_tests) * 100

    print(f"\n📈 ESTADÍSTICAS:")
    print(f"   Tests ejecutados: {total_tests}")
    print(f"   Tests exitosos: {passed_tests}")
    print(f"   Tests fallidos: {total_tests - passed_tests}")
    print(f"   Tasa de éxito: {success_rate:.1f}%")

    # 9. Verificaciones adicionales
    print(f"\n🔍 VERIFICACIONES ADICIONALES:")
    print(
        f"   📂 Archivos de datos: {'✅ Presentes' if data_files_present else '❌ Faltantes'}"
    )
    print(f"   🎮 CUDA disponible: {'✅ Sí' if torch.cuda.is_available() else '❌ No'}")
    if torch.cuda.is_available():
        print(f"   🎮 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   🐍 Versión Python: {sys.version.split()[0]}")
    print(f"   🔥 Versión PyTorch: {torch.__version__}")

    # 10. Plan de ejecución
    if time_estimate:
        print(f"\n{'='*70}")
        generate_execution_plan(time_estimate)
        print(f"{'='*70}")

    # 11. Recomendaciones finales
    print(f"\n💡 RECOMENDACIONES FINALES:")

    if success_rate == 100:
        print("   🎉 ¡Excelente! Todos los tests pasaron")
        print("   ✅ El proyecto está listo para ejecutarse")

        if data_files_present and time_estimate:
            total_hours = time_estimate["total_time"] / 3600
            if total_hours < 1:
                print("   🚀 Tiempo estimado < 1 hora - Ejecutar inmediatamente")
                print("   📝 Comando: python proyecto_deep_learning_dialogos.py")
            elif total_hours < 3:
                print(
                    f"   ⏰ Tiempo estimado ~{total_hours:.1f} horas - Planificar ejecución"
                )
                print("   📝 Comando: python proyecto_deep_learning_dialogos.py")
            else:
                print(
                    f"   🌙 Tiempo estimado ~{total_hours:.1f} horas - Ejecución nocturna recomendada"
                )
                print(
                    "   📝 Comando: nohup python proyecto_deep_learning_dialogos.py > output.log 2>&1 &"
                )
        else:
            print(
                "   📥 Descarga los archivos parquet para ejecutar el proyecto completo"
            )

    elif success_rate >= 80:
        print("   👍 La mayoría de tests pasaron")
        print("   🔧 Revisa los tests fallidos antes de ejecutar")
        print("   ⚠️ El proyecto podría funcionar con limitaciones")

    else:
        print("   ⚠️ Varios tests fallaron")
        print("   🔧 Revisa y corrige los errores antes de continuar")
        print("   ❌ No recomendado ejecutar el proyecto principal")

    # 12. Opciones adicionales
    print(f"\n🎯 OPCIONES DE EJECUCIÓN:")
    print(f"   🔬 Validación completa: python test_proyecto_completo.py")
    print(f"   🎮 Demos interactivos: python ejecutar_partes_especificas.py")
    print(f"   ⚡ Test rápido: python ejecutar_partes_especificas.py quick")
    print(f"   🧠 Solo modelos: python ejecutar_partes_especificas.py models")
    print(f"   📊 Solo datos: python ejecutar_partes_especificas.py data")

    # 13. Información de archivos de salida
    print(f"\n📁 ARCHIVOS QUE GENERARÁ EL PROYECTO:")
    expected_outputs = [
        "resultados_clasificacion_actos.json",
        "resultados_clasificacion_emociones.json",
        "analisis_rendimiento_actos.csv",
        "analisis_rendimiento_emociones.csv",
        "confusion_matrix_*.png",
        "training_history_*.png",
        "model_comparison_*.png",
        "architecture_*.png",
    ]

    for output in expected_outputs:
        print(f"   📄 {output}")

    # 14. Información de tiempo
    print(f"\n⏱️ INFORMACIÓN DE TIEMPO:")
    print(f"   Validación completada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if time_estimate:
        total_time = time_estimate["total_time"]
        print(f"   Proyecto completo estimado: {timedelta(seconds=total_time)}")

        # Mostrar horario recomendado
        now = datetime.now()
        estimated_finish = now + timedelta(seconds=total_time)
        print(
            f"   Si inicias ahora: Terminaría ~{estimated_finish.strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Sugerir horario nocturno si es muy largo
        if total_time > 7200:  # Más de 2 horas
            tonight = now.replace(hour=22, minute=0, second=0, microsecond=0)
            if tonight < now:
                tonight += timedelta(days=1)
            finish_tonight = tonight + timedelta(seconds=total_time)
            print(
                f"   Ejecución nocturna (22:00): Terminaría ~{finish_tonight.strftime('%Y-%m-%d %H:%M:%S')}"
            )

    # 15. Consejos de monitoreo
    if time_estimate and time_estimate["total_time"] > 1800:  # Más de 30 minutos
        print(f"\n👀 CONSEJOS DE MONITOREO:")
        print(f"   📊 El proyecto mostrará progreso en tiempo real")
        print(f"   💾 Los resultados se guardan automáticamente")
        print(f"   ⏹️ Puedes interrumpir con Ctrl+C si es necesario")
        print(
            f"   📝 Para ejecución en background: nohup python proyecto_deep_learning_dialogos.py > output.log 2>&1 &"
        )
        print(f"   👁️ Para monitorear: tail -f output.log")

    return success_rate >= 80


if __name__ == "__main__":
    success = main()

    print(f"\n{'='*70}")
    if success:
        print("🎯 VALIDACIÓN EXITOSA - El proyecto está listo")
        print("🚀 Puedes proceder con la ejecución")
        exit(0)
    else:
        print("⚠️ VALIDACIÓN CON PROBLEMAS - Revisa los errores")
        print("🔧 Corrige los problemas antes de ejecutar")
        exit(1)
