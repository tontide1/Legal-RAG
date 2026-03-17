def load_model(model_name):
    import subprocess

    # Load the specified model using OLLAMA
    try:
        subprocess.run(["ollama", "pull", model_name], check=True)
        print(f"Model {model_name} loaded successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error loading model {model_name}: {e}")

def infer(model_name, prompt):
    import subprocess

    # Run inference using the specified model and prompt
    try:
        result = subprocess.run(
            ["ollama", "run", model_name, prompt],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error during inference with model {model_name}: {e}")
        return None

def list_models():
    import subprocess

    # List available models in OLLAMA
    try:
        result = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True)
        return result.stdout.strip().splitlines()
    except subprocess.CalledProcessError as e:
        print(f"Error listing models: {e}")
        return []