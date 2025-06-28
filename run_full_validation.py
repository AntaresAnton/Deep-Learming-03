#!/usr/bin/env python3
"""
🎯 SCRIPT MAESTRO DE VALIDACIÓN
Ejecuta todas las validaciones en el orden correcto
"""

import os
import subprocess
import sys
from datetime import datetime


def run_command(command, description):
    """Ejecuta un comando y maneja errores"""
    print(f"\n{'='*60}")
    print(f"🔄 {description}")
    print(f"{'='*60}")

    try:
        if command.endswith(".py"):
            result = subprocess.run(
                [sys.executable, command], capture_output=False, text=True, check=True
            )
        else:
            result = subprocess.run(
                command, shell=True, capture_output=False, text=True, check=True
            )

        print(f"✅ {description} - COMPLETADO")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ {description} - FALLÓ")
        print(f"Código de error: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ {description} - ERROR: {e}")
        return False


def main():
    print("🎯 VALIDACIÓN COMPLETA DEL PROYECTO")
    print("=" * 60)
    print(f"⏰ Iniciado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Lista de validaciones a ejecutar
    validations = [
        ("check_data_files.py", "Verificación de archivos de datos"),
        ("test_quick_validation.py", "Validación rápida del código"),
    ]

    results = []

    for script, description in validations:
        if os.path.exists(script):
            success = run_command(script, description)
            results.append((description, success))
        else:
            print(f"⚠️ Script {script} no encontrado, saltando...")
            results.append((description, False))

    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN FINAL DE VALIDACIÓN")
    print(f"{'='*60}")

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for description, success in results:
        status = "✅ PASÓ" if success else "❌ FALLÓ"
        print(f"{status} - {description}")

    print(f"\n📈 Resultado: {passed}/{total} validaciones exitosas")
    print(f"⏰ Completado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if passed == total:
        print(f"\n🎉 ¡VALIDACIÓN COMPLETA EXITOSA!")
        print(f"✅ El proyecto está listo para ejecutarse")
        print(f"▶️ Puedes ejecutar el código principal con confianza")

        # Mostrar comando para ejecutar
        main_script = "proyecto_deep_learning_dialogos.py"
        if os.path.exists(main_script):
            print(f"\n🚀 Para ejecutar el proyecto completo:")
            print(f"   python {main_script}")

    else:
        print(f"\n⚠️ VALIDACIÓN INCOMPLETA")
        print(f"🔧 Soluciona los problemas indicados antes de continuar")

    return passed == total


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
