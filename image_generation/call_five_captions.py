import os
import openai
import joblib
from tqdm import tqdm

# Set your OpenAI API key here
openai.api_key = "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"

# File paths
file = "/Users/stanislav/Invisible-Relevance-Bias/flickr/Flickr30k/captions.txt"
new_file = "./flickr_merge/flickr30k_test_chatgpt_caps.txt"

# Create the directory if it doesn't exist
output_dir = os.path.dirname(new_file)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Open the .txt file and the output file
f = open(file, 'r')
f_write = open(new_file, 'w')

# List to store the final processed data
data_final = []

# Prompt to consolidate captions
prompt_template = (
    'Consolidate the five descriptions, avoid redundancy while including the scene described in each sentence, '
    'and make a concise summary as a prompt for text-to-image generation model to create an image. Avoid using '
    'the words "generate", "summary", and "prompt":\n'
)

# Limit the number of API requests to avoid hitting the quota quickly
MAX_REQUESTS = 10  # Set this value according to your quota/usage needs

# Process each line from the file
lines = f.readlines()

# Iterate over each line (with tqdm for progress tracking)
for idx, line in tqdm(enumerate(lines[:MAX_REQUESTS]), total=MAX_REQUESTS):
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
        'label': parts[1],  # Second part is the label
        'caption': []  # We will store the new consolidated caption here
    }
    
    # Captions (parts[2:] handles the case of commas in the caption)
    captions = parts[2:]

    # Format the text for the OpenAI prompt
    text = ""
    for i, caption in enumerate(captions):
        text += str(i + 1) + '.' + caption + '\n'
    text = prompt_template + text

    # Flag to ensure successful API call
    success_flag = 0

    # Retry mechanism in case of API failure
    while success_flag == 0:
        try:
            # Call OpenAI API (new API usage for gpt-3.5-turbo)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",  # Adjust engine/model as necessary
                messages=[
                    {"role": "user", "content": text}
                ],
                temperature=0,
                max_tokens=100,  # Adjust max_tokens if needed
                timeout=50
            )
            # Mark success
            success_flag = 1

            # Extract the generated text from the response
            new_text = response['choices'][0]['message']['content']
            new_one_data['caption'] = new_text  # Save the new caption
            
        except Exception as e:
            print(f"Request failed at line {idx}: {e}")
            success_flag = 0

    # Write the new processed data to the output file
    f_write.write(json.dumps(new_one_data) + "\n")
    
    # Append the new processed data to the final list
    data_final.append(new_one_data)
    
    # Save intermediate results
    joblib.dump(data_final, './flickr_merge/flickr30k_test_chatgpt_caps')

# Close files
f.close()
f_write.close()
python image_generation/call_five_captions.py