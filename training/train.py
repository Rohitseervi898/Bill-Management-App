# training/train.py

import subprocess
import sys

def train_model():
    """A simple wrapper for the 'spacy train' command."""
    
    config_path = "config.cfg"
    output_path = "../models/"
    
    # This is the command that will be executed in your terminal
    command = [
        sys.executable,  # Use the current python interpreter
        "-m", "spacy", "train",
        config_path,
        "--output", output_path,
        # Uncomment the line below and set a GPU ID if you have a compatible GPU
        # "--gpu-id", "0" 
    ]
    
    print("Executing command:")
    print(" ".join(command))
    
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("\n✅ Training complete!")
        print(f"Your trained model is saved in the '{output_path}' directory.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed with an error.")
        print(e)
    except FileNotFoundError:
        print("\n❌ Command failed. Is spaCy installed in your current environment?")
        print("Try running: pip install -U spacy")

if __name__ == "__main__":
    train_model()