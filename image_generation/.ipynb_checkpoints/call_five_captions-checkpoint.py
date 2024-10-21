import os
import joblib
import json
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCausalLM
import torch

# Load the LLaMA 3.2-11B Vision-Instruct model and processor from Hugging Face
model_name_or_path = "meta-llama/Llama-3.2-11B-Vision-Instruct"
processor = AutoProcessor.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path)

# Move the model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# File paths
file = "/Users/stanislav/Invisible-Relevance-Bias/flickr/Flickr30k/captions.txt"
new_file = "./flickr_merge/flickr30k_test_llama_caps.txt"

# Create the directory if it doesn't exist
output_dir = os.path.dirname(new_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the .txt file and the output file
with open(file, 'r') as f, open(new_file, 'w') as f_write:
    # List to store the final processed data
    data_final = []

    # Prompt to consolidate captions
    prompt_template = (
        'Consolidate the five descriptions, avoid redundancy while including the scene described in each sentence, '
        'and make a concise summary:\n'
    )

    # Read all lines from the file
    lines = f.readlines()

    # Iterate over each line (with tqdm for progress tracking)
    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        # Check if the data is already processed
        if idx < len(data_final):
            continue

        # Split the line into parts
        parts = line.strip().split(',')

        # Ensure that we have the correct format (image_name, label, and captions)
        if len(parts) < 3:
            print(f"Skipping line {idx} due to incorrect format")
            continue

        # Prepare data structure
        new_one_data = {
            'image_name': parts[0],  # First part is the image name
            'label': parts[1],       # Second part is the label
            'caption': []            # We will store the new consolidated caption here
        }

        # Captions (parts[2:] handles the case of commas in the caption)
        captions = parts[2:]

        # Format the text for the prompt
        text = prompt_template + "\n".join([f"{i + 1}. {caption}" for i, caption in enumerate(captions)])

        # Tokenize the input text for the model
        inputs = processor(text, return_tensors="pt").to(device)

        try:
            # Generate the caption using LLaMA
            outputs = model.generate(**inputs, max_length=100, temperature=0.7)
            new_text = processor.decode(outputs[0], skip_special_tokens=True)

            # Save the generated caption
            new_one_data['caption'] = new_text

        except Exception as e:
            print(f"Error processing line {idx}: {e}")
            continue  # Skip this line if there's an error

        # Write the new processed data to the output file
        f_write.write(json.dumps(new_one_data) + "\n")

        # Append the new processed data to the final list
        data_final.append(new_one_data)

        # Save intermediate results
        joblib.dump(data_final, './flickr_merge/flickr30k_test_llama_caps')

# Process completed
print("Processing completed and file saved.")
