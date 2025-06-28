#!/usr/bin/env python3
"""
🎯 EJECUTOR DE PARTES ESPECÍFICAS
Permite ejecutar solo componentes específicos del proyecto sin el entrenamiento completo
"""

import importlib.util
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch


def load_main_module():
    """Carga el módulo principal"""
    spec = importlib.util.spec_from_file_location(
        "main_project", "proyecto_deep_learning_dialogos.py"
    )
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    return main_module


def demo_tokenizer():
    """Demostración del tokenizador"""
    print("🔤 DEMO: SimpleTokenizer")
    print("-" * 40)

    main_module = load_main_module()

    # Textos de ejemplo
    texts = [
        "Hello, how are you today?",
        "I am doing great, thank you!",
        "What is your favorite color?",
        "I love blue and green colors",
        "Have a wonderful day!",
    ]

    print("📝 Textos de entrada:")
    for i, text in enumerate(texts, 1):
        print(f"   {i}. {text}")

    # Crear tokenizador
    tokenizer = main_module.SimpleTokenizer(texts, vocab_size=50)

    print(f"\n📚 Vocabulario creado: {tokenizer.vocab_size} palabras")
    print(f"🔤 Primeras 10 palabras del vocabulario:")

    for i, (word, idx) in enumerate(list(tokenizer.word_to_idx.items())[:10]):
        print(f"   {idx:2d}: {word}")

    # Tokenizar un ejemplo
    example_text = "Hello, how are you?"
    words = example_text.lower().split()
    tokens = [
        tokenizer.word_to_idx.get(word, tokenizer.word_to_idx["<UNK>"])
        for word in words
    ]

    print(f"\n🔍 Ejemplo de tokenización:")
    print(f"   Texto: '{example_text}'")
    print(f"   Palabras: {words}")
    print(f"   Tokens: {tokens}")


def demo_models():
    """Demostración de los modelos"""
    print("🧠 DEMO: Modelos de Deep Learning")
    print("-" * 40)

    main_module = load_main_module()

    # Parámetros
    vocab_size = 1000
    embedding_dim = 64
    hidden_dim = 128
    output_dim = 5
    batch_size = 3
    seq_length = 15

    # Datos dummy
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)

    print(f"📊 Configuración:")
    print(f"   Tamaño del vocabulario: {vocab_size}")
    print(f"   Dimensión de embedding: {embedding_dim}")
    print(f"   Dimensión oculta: {hidden_dim}")
    print(f"   Clases de salida: {output_dim}")
    print(f"   Tamaño del batch: {batch_size}")
    print(f"   Longitud de secuencia: {seq_length}")

    # Probar cada modelo
    models = [
        (
            "SimpleRNN",
            main_module.SimpleRNN(vocab_size, embedding_dim, hidden_dim, output_dim),
        ),
        (
            "LSTM",
            main_module.LSTMClassifier(
                vocab_size, embedding_dim, hidden_dim, output_dim
            ),
        ),
        (
            "GRU",
            main_module.GRUClassifier(
                vocab_size, embedding_dim, hidden_dim, output_dim
            ),
        ),
        (
            "Transformer",
            main_module.TransformerClassifier(vocab_size, embedding_dim, output_dim),
        ),
    ]

    print(f"\n🔍 Probando modelos:")

    for model_name, model in models:
        print(f"\n   🧠 {model_name}:")

        # Contar parámetros
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"      Parámetros totales: {total_params:,}")
        print(f"      Parámetros entrenables: {trainable_params:,}")

        # Forward pass
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            print(f"      Forma de salida: {output.shape}")
            print(f"      Rango de valores: [{output.min():.3f}, {output.max():.3f}]")


def demo_data_processing():
    """Demostración del procesamiento de datos"""
    print("📊 DEMO: Procesamiento de Datos")
    print("-" * 40)

    main_module = load_main_module()

    # Crear datos dummy que simulen la estructura real
    dummy_data = pd.DataFrame(
        {
            "dialog": [
                ["Hello there!", "How are you doing today?", "I am fine, thanks"],
                ["Good morning", "Nice weather we are having", "Yes, very pleasant"],
                ["Thank you so much", "You are very welcome", "Have a great day"],
            ],
            "act": [
                ["greeting", "question", "inform"],
                ["greeting", "inform", "agree"],
                ["thanking", "acknowledge", "closing"],
            ],
            "emotion": [
                ["neutral", "curious", "happy"],
                ["happy", "happy", "happy"],
                ["grateful", "kind", "happy"],
            ],
        }
    )

    print("📝 Datos originales:")
    print(dummy_data.to_string(index=False))

    # Expandir diálogos
    expanded = main_module.expand_dialogues(dummy_data)

    print(f"\n🔄 Datos expandidos:")
    print(f"   Filas originales: {len(dummy_data)}")
    print(f"   Filas expandidas: {len(expanded)}")
    print("\n📋 Primeras 5 filas expandidas:")
    print(expanded.head().to_string(index=False))

    # Estadísticas
    print(f"\n📊 Estadísticas:")
    print(f"   Actos únicos: {expanded['act'].nunique()}")
    print(f"   Emociones únicas: {expanded['emotion'].nunique()}")
    print(
        f"   Longitud promedio de diálogo: {expanded['dialog'].str.len().mean():.1f} caracteres"
    )

    print(f"\n🎭 Distribución de actos:")
    act_counts = expanded["act"].value_counts()
    for act, count in act_counts.items():
        print(f"   {act}: {count}")

    print(f"\n😊 Distribución de emociones:")
    emotion_counts = expanded["emotion"].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion}: {count}")


def demo_training_step():
    """Demostración de un paso de entrenamiento"""
    print("🏋️ DEMO: Paso de Entrenamiento")
    print("-" * 40)

    main_module = load_main_module()

    # Crear datos dummy
    texts = [
        "hello world",
        "this is great",
        "machine learning rocks",
        "deep learning works",
    ]
    labels = [0, 1, 0, 1]

    # Crear tokenizador
    tokenizer = main_module.SimpleTokenizer(texts, vocab_size=50)

    # Crear dataset
    dataset = main_module.DialogDataset(
        texts, labels, tokenizer.word_to_idx, max_length=10
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=True)

    # Crear modelo
    model = main_module.SimpleRNN(
        vocab_size=tokenizer.vocab_size, embedding_dim=32, hidden_dim=64, output_dim=2
    )

    # Configurar entrenamiento
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    print(f"📊 Configuración del entrenamiento:")
    print(f"   Modelo: SimpleRNN")
    print(f"   Optimizador: Adam (lr=0.01)")
    print(f"   Función de pérdida: CrossEntropyLoss")
    print(f"   Tamaño del batch: 2")
    print(f"   Número de muestras: {len(dataset)}")

    # Ejecutar algunos pasos de entrenamiento
    model.train()
    print(f"\n🔄 Ejecutando 3 pasos de entrenamiento:")

    for step, batch in enumerate(dataloader):
        if step >= 3:  # Solo 3 pasos
            break

        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]

        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calcular accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels).float().sum()
        accuracy = correct / len(labels)

        print(
            f"   Paso {step + 1}: Pérdida = {loss.item():.4f}, Accuracy = {accuracy.item():.4f}"
        )

    print(f"✅ Demostración de entrenamiento completada")


def demo_evaluation():
    """Demostración de evaluación"""
    print("📊 DEMO: Evaluación de Modelo")
    print("-" * 40)

    main_module = load_main_module()

    # Simular resultados de evaluación
    dummy_results = {
        "SimpleRNN": {
            "accuracy": 0.75,
            "precision": 0.73,
            "recall": 0.72,
            "f1": 0.74,
            "classification_report": {
                "0": {"precision": 0.8, "recall": 0.7, "f1-score": 0.75},
                "1": {"precision": 0.7, "recall": 0.8, "f1-score": 0.74},
            },
        },
        "LSTM": {
            "accuracy": 0.82,
            "precision": 0.81,
            "recall": 0.80,
            "f1": 0.81,
            "classification_report": {
                "0": {"precision": 0.85, "recall": 0.78, "f1-score": 0.81},
                "1": {"precision": 0.78, "recall": 0.85, "f1-score": 0.81},
            },
        },
        "GRU": {
            "accuracy": 0.79,
            "precision": 0.78,
            "recall": 0.77,
            "f1": 0.78,
            "classification_report": {
                "0": {"precision": 0.82, "recall": 0.75, "f1-score": 0.78},
                "1": {"precision": 0.75, "recall": 0.82, "f1-score": 0.78},
            },
        },
        "Transformer": {
            "accuracy": 0.85,
            "precision": 0.84,
            "recall": 0.83,
            "f1": 0.84,
            "classification_report": {
                "0": {"precision": 0.87, "recall": 0.81, "f1-score": 0.84},
                "1": {"precision": 0.81, "recall": 0.87, "f1-score": 0.84},
            },
        },
    }

    print("📈 Resultados simulados de evaluación:")
    print("-" * 60)
    print(
        f"{'Modelo':<12} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}"
    )
    print("-" * 60)

    for model_name, results in dummy_results.items():
        print(
            f"{model_name:<12} {results['accuracy']:<10.3f} {results['precision']:<10.3f} "
            f"{results['recall']:<10.3f} {results['f1']:<10.3f}"
        )

    # Encontrar el mejor modelo
    best_model = max(dummy_results.keys(), key=lambda x: dummy_results[x]["f1"])
    best_f1 = dummy_results[best_model]["f1"]

    print(f"\n🏆 Mejor modelo: {best_model} (F1-Score: {best_f1:.3f})")

    # Análisis detallado
    print(f"\n🔍 Análisis detallado:")
    analysis_df = main_module.analyze_model_performance(dummy_results, "Demo Task")

    return analysis_df


def demo_visualizations():
    """Demostración de visualizaciones"""
    print("📊 DEMO: Visualizaciones")
    print("-" * 40)

    main_module = load_main_module()

    print("🎨 Generando visualizaciones de arquitecturas...")

    # Visualizar cada arquitectura
    architectures = ["SimpleRNN", "LSTM", "GRU", "Transformer"]

    for arch in architectures:
        print(f"   📐 Visualizando {arch}...")
        try:
            main_module.visualize_model_architecture(arch)
            print(f"      ✅ {arch} visualizado correctamente")
        except Exception as e:
            print(f"      ❌ Error visualizando {arch}: {e}")

    print(f"\n📈 Las visualizaciones se muestran en ventanas separadas")
    print(f"💡 Cierra las ventanas para continuar")


def demo_hyperparameter_analysis():
    """Demostración del análisis de hiperparámetros"""
    print("🔧 DEMO: Análisis de Hiperparámetros")
    print("-" * 40)

    main_module = load_main_module()

    # Ejecutar análisis de hiperparámetros
    main_module.hyperparameter_analysis()

    print(f"\n💡 Recomendaciones adicionales:")
    print(f"   • Usar learning rate scheduling para mejor convergencia")
    print(f"   • Implementar gradient clipping para modelos RNN")
    print(f"   • Considerar data augmentation para datasets pequeños")
    print(f"   • Usar early stopping para prevenir overfitting")


def interactive_menu():
    """Menú interactivo para seleccionar demos"""
    print("🎯 MENÚ INTERACTIVO - DEMOS DEL PROYECTO")
    print("=" * 50)

    options = {
        "1": ("Tokenizador", demo_tokenizer),
        "2": ("Modelos", demo_models),
        "3": ("Procesamiento de Datos", demo_data_processing),
        "4": ("Paso de Entrenamiento", demo_training_step),
        "5": ("Evaluación", demo_evaluation),
        "6": ("Visualizaciones", demo_visualizations),
        "7": ("Análisis de Hiperparámetros", demo_hyperparameter_analysis),
        "8": ("Ejecutar Todos", None),
        "0": ("Salir", None),
    }

    while True:
        print(f"\n📋 Opciones disponibles:")
        for key, (name, _) in options.items():
            print(f"   {key}. {name}")

        choice = input(f"\n🔍 Selecciona una opción (0-8): ").strip()

        if choice == "0":
            print("👋 ¡Hasta luego!")
            break
        elif choice == "8":
            print("🚀 Ejecutando todas las demos...")
            for key in ["1", "2", "3", "4", "5", "6", "7"]:
                print(f"\n{'='*60}")
                try:
                    options[key][1]()
                except Exception as e:
                    print(f"❌ Error en demo {options[key][0]}: {e}")
                print(f"{'='*60}")
            print("✅ Todas las demos completadas")
        elif choice in options and choice != "0" and choice != "8":
            print(f"\n{'='*60}")
            try:
                options[choice][1]()
            except Exception as e:
                print(f"❌ Error ejecutando {options[choice][0]}: {e}")
            print(f"{'='*60}")
        else:
            print("❌ Opción no válida. Intenta de nuevo.")


def quick_test():
    """Test rápido de todos los componentes"""
    print("⚡ TEST RÁPIDO DE COMPONENTES")
    print("=" * 40)

    components = [
        ("Tokenizador", demo_tokenizer),
        ("Modelos", demo_models),
        ("Procesamiento", demo_data_processing),
        ("Entrenamiento", demo_training_step),
        ("Evaluación", demo_evaluation),
    ]

    results = []

    for name, func in components:
        print(f"\n🔍 Probando {name}...")
        try:
            func()
            print(f"✅ {name} - OK")
            results.append((name, True))
        except Exception as e:
            print(f"❌ {name} - ERROR: {e}")
            results.append((name, False))

    # Resumen
    print(f"\n📊 RESUMEN DEL TEST RÁPIDO:")
    print("-" * 40)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅" if success else "❌"
        print(f"{status} {name}")

    print(f"\n📈 Resultado: {passed}/{total} componentes funcionando")

    if passed == total:
        print("🎉 ¡Todos los componentes funcionan correctamente!")
    elif passed >= total * 0.8:
        print("👍 La mayoría de componentes funcionan bien")
    else:
        print("⚠️ Varios componentes tienen problemas")


def main():
    """Función principal"""
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

        if mode == "quick":
            quick_test()
        elif mode == "tokenizer":
            demo_tokenizer()
        elif mode == "models":
            demo_models()
        elif mode == "data":
            demo_data_processing()
        elif mode == "train":
            demo_training_step()
        elif mode == "eval":
            demo_evaluation()
        elif mode == "viz":
            demo_visualizations()
        elif mode == "hyper":
            demo_hyperparameter_analysis()
        elif mode == "all":
            quick_test()
        else:
            print(f"❌ Modo '{mode}' no reconocido")
            print(
                "💡 Modos disponibles: quick, tokenizer, models, data, train, eval, viz, hyper, all"
            )
    else:
        interactive_menu()


if __name__ == "__main__":
    print(f"🚀 EJECUTOR DE PARTES ESPECÍFICAS")
    print(f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎮 Dispositivo: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print()

    try:
        main()
    except KeyboardInterrupt:
        print(f"\n⏹️ Interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback

        traceback.print_exc()
