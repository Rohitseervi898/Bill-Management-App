import spacy
import json

# 1. Define the path to your custom-trained model
#    This should point to the "model-best" folder created by the training process.
MODEL_PATH = "./models/model-best"

# 2. (Optional) Load a sample of OCR text from one of your JSON files to test
#    Replace 'invoice1.json' with the name of any file in your 'output_json' folder.
try:
    with open("./output_json/invoice3.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        sample_text = data['data']['text']
except FileNotFoundError:
    print("Test JSON file not found. Using default sample text.")
    # Fallback text if the JSON file isn't found
    sample_text = """
    SHREE CHARBHUJA STORE
    Tax Invoice No: GEN-7170
    Invoice Date :10/02/2024
    Net Payable : 4886.00
    """

# 3. Load your custom spaCy model
print(f"Loading model from: {MODEL_PATH}")
try:
    nlp = spacy.load(MODEL_PATH)
    print("✅ Model loaded successfully!")
except OSError:
    print(f"❌ Error: Could not find a model at '{MODEL_PATH}'.")
    print("Please make sure you have successfully run 'python train.py' and the 'model-best' folder exists.")
    exit()

# 4. Process the sample text with your model
print("\n--- Processing Text ---")
print(sample_text)
doc = nlp(sample_text)

# 5. Print the entities found by your model
print("\n--- Entities Found ---")
if doc.ents:
    for ent in doc.ents:
        # Print the extracted text and the label your model assigned to it
        print(f"  -> Entity: '{ent.text}',  Label: '{ent.label_}'")
else:
    print("No entities found in the sample text.")