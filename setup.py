#!/usr/bin/env python3
"""
Setup script for Name Screening Tool.
Downloads required spaCy models and initializes the system.
"""

import subprocess
import sys


def download_spacy_models():
    """Download required spaCy models."""
    models = [
        "en_core_web_sm",  # English model for entity extraction
        "en_core_web_md",  # Medium model with word vectors (optional)
    ]
    
    print("🔄 Downloading spaCy models...")
    print("=" * 60)
    
    for model in models:
        print(f"\n📦 Downloading {model}...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "spacy", "download", model],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            print(f"✅ {model} downloaded successfully!")
        except subprocess.CalledProcessError:
            # Try with poetry run if direct call fails
            try:
                subprocess.check_call(
                    ["poetry", "run", "python", "-m", "spacy", "download", model]
                )
                print(f"✅ {model} downloaded successfully!")
            except Exception as e:
                print(f"❌ Failed to download {model}: {e}")
                print(f"   Please run manually: python -m spacy download {model}")
    
    print("\n" + "=" * 60)
    print("✅ Setup complete! You can now use the name screening tool.")
    print("\nQuick start:")
    print("  poetry run python -m src.name_screening.main --help")
    print("  poetry run python web_app.py")


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import spacy
        import pydantic
        import typer
        import textdistance
        print("✅ All Python dependencies are installed.")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please run: poetry install")
        return False


def main():
    """Main setup function."""
    print("Name Screening Tool - Setup")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Download spaCy models
    download_spacy_models()


if __name__ == "__main__":
    main()