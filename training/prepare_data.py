import spacy
import srsly
from tqdm import tqdm
from spacy.tokens import DocBin

def convert_label_studio_to_spacy(filepath: str, output_path: str):
    """
    Converts a JSON export from Label Studio into a .spacy file for training,
    correctly parsing the nested data structure.
    """
    nlp = spacy.blank("en")
    db = DocBin()

    try:
        training_data = list(srsly.read_json(filepath))
        print(f"Loaded {len(training_data)} examples from {filepath}")
    except Exception as e:
        print(f"Error loading or parsing JSON file: {e}")
        return

    for example in tqdm(training_data, desc="Processing data"):
        try:
            # CORRECTED: Access the text inside the 'data' object
            text = example['data']['text']
            doc = nlp.make_doc(text)
            ents = []

            # CORRECTED: Loop through the nested annotation structure
            for annotation in example.get('annotations', []):
                for result in annotation.get('result', []):
                    # Ensure the result is for labels and has a 'value'
                    if 'value' in result and result['type'] == 'labels':
                        value = result['value']
                        start = value['start']
                        end = value['end']
                        # The label is in a list, so we take the first one
                        label = value['labels'][0]

                        span = doc.char_span(start, end, label=label, alignment_mode="contract")

                        if span is None:
                            print(f"\nWarning: Skipping entity because span is invalid. Text: '{text[start:end]}'")
                        else:
                            ents.append(span)
            
            doc.ents = ents
            db.add(doc)

        except KeyError as e:
            print(f"\nWarning: Skipping example due to missing key: {e}. Check your JSON structure.")
            print(f"Problematic example (first 100 chars): {str(example)[:100]}...")
        except Exception as e:
            print(f"\nAn error occurred while processing an example: {e}")

    db.to_disk(output_path)
    print(f"\nSuccessfully created {output_path}")
    print("You can now run 'python train.py' to start training your model.")

if __name__ == '__main__':
    input_file = "../data/labeled_data.json"
    output_file = "./train.spacy"  # Save it directly in the training folder

    convert_label_studio_to_spacy(input_file, output_file)