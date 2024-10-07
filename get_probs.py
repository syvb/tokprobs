from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import re
from flask import Flask, send_file, jsonify, request, make_response
import os

app = Flask(__name__)

def clean_token(token):
    replacements = [
        ('Ġ', ' '),
        ('Ċ', '\n'),
    ]
    
    for old, new in replacements:
        token = token.replace(old, new)
    
    token = re.sub(r'\\u[0-9a-fA-F]{4}', lambda m: chr(int(m.group(0)[2:], 16)), token)
    token = token.lstrip()
    
    return token

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Get the output embeddings
output_embeddings = model.get_output_embeddings().weight.detach()

# Perform SVD to reduce dimensions to 2
U, S, V = torch.svd(output_embeddings)
projection_matrix = V[:, :2]

# Create a cleaned token to ID mapping
token_to_id = {clean_token(token): id for token, id in tokenizer.get_vocab().items()}

def generate_js_content():
    reduced_embeddings = torch.mm(output_embeddings, projection_matrix).cpu().numpy()
    return f"""
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

# Generate JS content on start
js_content = generate_js_content()

@app.route('/')
def serve_html():
    return send_file('index.html')

@app.route('/output_embeddings.js')
def serve_js():
    response = make_response(js_content)
    response.headers['Content-Type'] = 'application/javascript'
    response.headers['Cache-Control'] = 'no-cache'
    return response

@app.route('/get_embedding', methods=['POST'])
def get_embedding():
    text = request.json['text']
    input_ids = tokenizer.encode(text, return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        # Get the output from the last hidden layer
        outputs = model(input_ids, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        
        # Get the vector for the last token
        last_token_vector = last_hidden_state[0, -1, :]
        
        # Normalize the vector
        normalized_vector = last_token_vector / torch.norm(last_token_vector)
        
        # Project the normalized vector to 2D space
        reduced_vector = torch.mm(normalized_vector.unsqueeze(0), projection_matrix)
        
    embedding = reduced_vector.squeeze().cpu().numpy().tolist()
    return jsonify({"embedding": embedding})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
