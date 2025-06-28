#!/usr/bin/env python3
"""
📂 VERIFICADOR DE ARCHIVOS DE DATOS
Verifica que los archivos parquet estén presentes y tengan la estructura correcta
"""

import os
import sys
from pathlib import Path

import pandas as pd


def check_file_exists(filename):
    """Verifica si un archivo existe"""
    if os.path.exists(filename):
        size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f"✅ {filename} - {size:.1f} MB")
        return True
    else:
        print(f"❌ {filename} - NO ENCONTRADO")
        return False


def check_data_structure(filename):
    """Verifica la estructura de los datos"""
    try:
        df = pd.read_parquet(filename)
        print(f"   📊 Forma: {df.shape}")
        print(f"   📋 Columnas: {list(df.columns)}")

        # Verificar columnas esperadas
        expected_cols = ["dialog", "act", "emotion"]
        missing_cols = [col for col in expected_cols if col not in df.columns]

        if missing_cols:
            print(f"   ⚠️ Columnas faltantes: {missing_cols}")
            return False

        # Verificar tipos de datos
        for col in expected_cols:
            sample_value = df[col].iloc[0]
            if isinstance(sample_value, list):
                print(f"   ✅ {col}: Lista con {len(sample_value)} elementos")
            else:
                print(f"   ⚠️ {col}: Tipo inesperado - {type(sample_value)}")

        return True

    except Exception as e:
        print(f"   ❌ Error al leer archivo: {e}")
        return False


def main():
    print("📂 VERIFICADOR DE ARCHIVOS DE DATOS")
    print("=" * 50)

    # Archivos requeridos
    required_files = ["train.parquet", "validation.parquet", "test.parquet"]

    all_present = True
    all_valid = True

    for filename in required_files:
        print(f"\n🔍 Verificando {filename}:")

        if check_file_exists(filename):
            if not check_data_structure(filename):
                all_valid = False
        else:
            all_present = False
            all_valid = False

    # Resumen
    print(f"\n{'='*50}")
    print("📊 RESUMEN:")

    if all_present and all_valid:
        print("✅ Todos los archivos están presentes y son válidos")
        print("🚀 Puedes proceder con el entrenamiento")
        return True
    elif all_present:
        print("⚠️ Archivos presentes pero con problemas de estructura")
        print("🔧 Revisa los warnings arriba")
        return False
    else:
        print("❌ Archivos faltantes")
        print("📥 Descarga los archivos parquet necesarios:")
        for f in required_files:
            if not os.path.exists(f):
                print(f"   - {f}")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
