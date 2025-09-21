# training/prepare_data.py

import spacy
import srsly  # a helpful library for reading/writing files, comes with spaCy
from tqdm import tqdm # for a nice progress bar
from spacy.tokens import DocBin

def convert_label_studio_to_spacy(filepath: str, output_path: str):
    """
    Converts a JSON export from Label Studio into a .spacy file for training.
    """
    # Load a blank English model. We only need its tokenizer.
    nlp = spacy.blank("en")
    
    # This will store our processed documents
    db = DocBin()

    # Load the exported data from Label Studio
    training_data = list(srsly.read_json(filepath))
    print(f"Loaded {len(training_data)} examples from {filepath}")

    for example in tqdm(training_data, desc="Processing data"):
        text = example['text']
        annotations = example.get('label', []) # 'label' is the default key in Label Studio
        
        doc = nlp.make_doc(text)
        ents = []
        
        for ann in annotations:
            # Get the start, end, and label name
            start = ann['start']
            end = ann['end']
            label = ann['label'] # This is the entity label, e.g., "INVOICE_NUMBER"
            
            # Create a spaCy Span object. We must ensure the label is valid.
            span = doc.char_span(start, end, label=label, alignment_mode="contract")
            
            if span is None:
                print(f"Skipping entity: Span for '{text[start:end]}' not found in doc.")
            else:
                ents.append(span)
        
        try:
            doc.ents = ents
            db.add(doc)
        except ValueError as e:
            print(f"Error setting entities for doc: {text[:50]}... | Error: {e}")


    # Save the DocBin to the specified output path
    db.to_disk(output_path)
    print(f"\nSuccessfully created {output_path}")

if __name__ == '__main__':
    # Define the input and output file paths
    # IMPORTANT: You will replace 'labeled_data.json' with the actual name
    # of the file you export from Label Studio.
    input_file = "../data/labeled_data.json"
    output_file = "../training/train.spacy"

    convert_label_studio_to_spacy(input_file, output_file)
    
    # We can do the same for a validation set if we have one
    # convert_label_studio_to_spacy("../data/validation_data.json", "../training/dev.spacy")