#!/usr/bin/env python3
"""
🎨 FORMATEADOR DE PROYECTO PYTHON
Script simple para formatear e indentar código Python automáticamente
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Ejecuta un comando y muestra el resultado"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completado")
            return True
        else:
            print(f"❌ Error en {description}: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error ejecutando {description}: {e}")
        return False


def format_python_files(directory=".", config_type="pep8"):
    """Formatea todos los archivos Python en el directorio"""

    print(f"🎨 FORMATEADOR DE PROYECTO PYTHON")
    print("=" * 50)
    print(f"📁 Directorio: {os.path.abspath(directory)}")
    print(f"⚙️ Configuración: {config_type}")
    print("=" * 50)

    # Buscar archivos Python
    python_files = list(Path(directory).rglob("*.py"))

    if not python_files:
        print("❌ No se encontraron archivos Python")
        return

    print(f"📄 Archivos encontrados: {len(python_files)}")
    for file in python_files:
        print(f"   📝 {file}")

    print("\n🚀 Iniciando formateo...")

    # 1. Organizar imports con isort
    print("\n1️⃣ ORGANIZANDO IMPORTS")
    isort_cmd = f"isort {directory} --profile=black"
    run_command(isort_cmd, "Organizando imports")

    # 2. Formatear con Black o autopep8 según configuración
    if config_type == "black":
        print("\n2️⃣ FORMATEANDO CON BLACK")
        black_cmd = f"black {directory} --line-length=88"
        run_command(black_cmd, "Formateando con Black")

    elif config_type == "pep8":
        print("\n2️⃣ FORMATEANDO CON AUTOPEP8")
        autopep8_cmd = (
            f"autopep8 --in-place --recursive --aggressive --aggressive {directory}"
        )
        run_command(autopep8_cmd, "Formateando con autopep8")

    # 3. Verificar sintaxis
    print("\n3️⃣ VERIFICANDO SINTAXIS")
    syntax_errors = []

    for file in python_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                compile(f.read(), file, "exec")
            print(f"✅ {file.name}")
        except SyntaxError as e:
            syntax_errors.append((file, e))
            print(f"❌ {file.name}: {e}")

    # Resumen final
    print("\n" + "=" * 50)
    print("📊 RESUMEN DEL FORMATEO")
    print("=" * 50)
    print(f"📄 Archivos procesados: {len(python_files)}")
    print(f"✅ Archivos sin errores: {len(python_files) - len(syntax_errors)}")
    print(f"❌ Archivos con errores: {len(syntax_errors)}")

    if syntax_errors:
        print("\n⚠️ ERRORES DE SINTAXIS:")
        for file, error in syntax_errors:
            print(f"   📝 {file}: Línea {error.lineno} - {error.msg}")

    print(f"\n🎉 ¡Formateo completado!")


def create_config_files(directory="."):
    """Crea archivos de configuración para herramientas de formateo"""

    print(f"⚙️ Creando archivos de configuración...")

    # pyproject.toml para Black e isort
    pyproject_content = """[tool.black]
line-length = 88
target-version = ['py38']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
multi_line_output = 3
line_length = 88
"""

    # .flake8 para flake8
    flake8_content = """[flake8]
max-line-length = 88
extend-ignore = E203, E266, E501, W503
max-complexity = 10
select = B,C,E,F,W,T4,B9
"""

    # setup.cfg para autopep8
    setup_cfg_content = """[pycodestyle]
max-line-length = 88
ignore = E203,E266,E501,W503

[autopep8]
max-line-length = 88
aggressive = 2
"""

    configs = {
        "pyproject.toml": pyproject_content,
        ".flake8": flake8_content,
        "setup.cfg": setup_cfg_content,
    }

    for filename, content in configs.items():
        filepath = Path(directory) / filename
        if not filepath.exists():
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✅ Creado: {filename}")
        else:
            print(f"⚠️ Ya existe: {filename}")


def install_dependencies():
    """Instala las dependencias necesarias"""

    print(f"📦 Instalando dependencias...")

    dependencies = ["black", "isort", "autopep8", "flake8"]

    for dep in dependencies:
        cmd = f"pip install {dep}"
        if run_command(cmd, f"Instalando {dep}"):
            print(f"✅ {dep} instalado")
        else:
            print(f"❌ Error instalando {dep}")


def main():
    """Función principal"""

    parser = argparse.ArgumentParser(description="Formateador de proyecto Python")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directorio a formatear (default: directorio actual)",
    )
    parser.add_argument(
        "--config",
        choices=["black", "pep8"],
        default="black",
        help="Tipo de configuración (default: black)",
    )
    parser.add_argument(
        "--install", action="store_true", help="Instalar dependencias necesarias"
    )
    parser.add_argument(
        "--setup", action="store_true", help="Crear archivos de configuración"
    )
    parser.add_argument(
        "--check", action="store_true", help="Solo verificar, no formatear"
    )

    args = parser.parse_args()

    # Verificar que el directorio existe
    if not os.path.exists(args.directory):
        print(f"❌ El directorio '{args.directory}' no existe")
        sys.exit(1)

    # Instalar dependencias si se solicita
    if args.install:
        install_dependencies()
        return

    # Crear archivos de configuración si se solicita
    if args.setup:
        create_config_files(args.directory)
        return

    # Verificar solo si se solicita
    if args.check:
        print(f"🔍 Verificando formato en {args.directory}...")
        if args.config == "black":
            run_command(f"black --check {args.directory}", "Verificando con Black")
        else:
            run_command(
                f"autopep8 --diff --recursive {args.directory}",
                "Verificando con autopep8",
            )
        return

    # Formatear archivos
    format_python_files(args.directory, args.config)


if __name__ == "__main__":
    main()
