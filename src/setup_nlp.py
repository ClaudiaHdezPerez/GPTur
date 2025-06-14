import subprocess
import sys

def setup_nlp():
    print("Installing required NLP models...")
    
    try:
        # Install Spanish language model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "es_core_news_md"])
        print("Successfully installed Spanish language model")
        
    except subprocess.CalledProcessError as e:
        print(f"Error installing models: {str(e)}")
        sys.exit(1)
    
    print("NLP setup completed successfully!")

if __name__ == "__main__":
    setup_nlp()
