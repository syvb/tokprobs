from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re

def clean_token(token):
    # Define replacement rules
    replacements = [
        ('Ġ', ' '),  # Space at the beginning of a word
        ('Ċ', '\n'),  # Newline
        #('ĉ', '\r'),  # Carriage return
        #('Ń', ' '),  # Space (less common representation)
        #('Ă', '\t'),  # Tab
        #('Ġ', ' '),  # Space (alternative representation)
        #('Ė', ' '),  # Space (another alternative)
        #('Ơ', ''),   # Empty string (remove this token)
    ]
    
    # Apply replacements
    for old, new in replacements:
        token = token.replace(old, new)
    
    # Handle hex-encoded characters (e.g., \u0120)
    token = re.sub(r'\\u[0-9a-fA-F]{4}', lambda m: chr(int(m.group(0)[2:], 16)), token)
    
    # Remove leading space if present
    token = token.lstrip()
    
    return token

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Get the output embeddings
output_embeddings = model.get_output_embeddings().weight.detach()

# Perform SVD to reduce dimensions to 2
U, S, V = torch.svd(output_embeddings)
reduced_embeddings = torch.mm(output_embeddings, V[:, :2])

# Convert to numpy for easier handling
reduced_embeddings = reduced_embeddings.cpu().numpy()

# Create a cleaned token to ID mapping
token_to_id = {clean_token(token): id for token, id in tokenizer.get_vocab().items()}

# Create the JavaScript content
js_content = f"""
// Token to ID mapping
const tokenToId = {json.dumps(token_to_id)};

// Embeddings as Float32Array
const embeddings = new Float32Array([
  {','.join(map(str, reduced_embeddings.flatten()))}
]);

// Function to get embeddings for a given token string or ID
function getEmbedding(tokenOrId) {{
  let id = typeof tokenOrId === 'string' ? tokenToId[tokenOrId] : tokenOrId;
  if (id === undefined || id < 0 || id >= {len(token_to_id)}) {{
    throw new Error('Invalid token or ID');
  }}
  return [embeddings[id * 2], embeddings[id * 2 + 1]];
}}
"""

# Write the JavaScript file
with open('output_embeddings.js', 'w') as f:
    f.write(js_content)

print("JavaScript file 'output_embeddings.js' has been created successfully.")