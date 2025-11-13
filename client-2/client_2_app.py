import numpy as np
from flask import Flask, request, jsonify

# --- Hyperparameters ---
LEARNING_RATE = 0.01
LOCAL_EPOCHS = 10  # E: Number of training steps each client takes

# --- Client 2's Private Data (n_k = 2) ---
# (Points are around y = 1x + 1)
MY_DATA = {
    'x': np.array([3.0, 4.0]), 
    'y': np.array([4.0, 5.0])
}
DATA_SIZE = len(MY_DATA['x']) # n_k

# Create the Flask App
app = Flask(__name__)

# =============================================================================
# The Local Training Logic (The "How to Train" Math)
# =================================S============================================
def train_local_model(global_weights, data):
    """Trains a model on local data using raw gradient descent."""
    
    # Start from the server's global model
    w = global_weights['w']
    b = global_weights['b']
    
    n = len(data['x']) # Number of data points
    
    print(f"  Starting local training with global model: w={w:.4f}, b={b:.4f}")

    # Train for E local epochs
    for e in range(LOCAL_EPOCHS):
        # 1. Make predictions
        y_pred = w * data['x'] + b
        
        # 2. Calculate error (Loss)
        error = y_pred - data['y']
        
        # 3. Calculate Gradients (The raw math)
        # dL/dw = (2/n) * sum(x * error)
        # dL/db = (2/n) * sum(error)
        grad_w = (2/n) * np.sum(data['x'] * error)
        grad_b = (2/n) * np.sum(error)
        
        # 4. Update local model (Gradient Descent step)
        w = w - LEARNING_RATE * grad_w
        b = b - LEARNING_RATE * grad_b
    
    print(f"  Finished local training. New local model: w={w:.4f}, b={b:.4f}")
    
    # 5. Return the new local model weights
    return {'w': w, 'b': b}

# =============================================================================
# The API Endpoint for the Server
# =============================================================================
@app.route('/train', methods=['POST'])
def handle_train_request():
    """
    This is the main API endpoint. The Server will call this.
    It receives the global model, trains it locally, and returns
    the new local model and this client's data size.
    """
    try:
        # 1. Get the server's current global model from the request
        request_data = request.json
        global_model = request_data['global_model']
        
        print(f"\n[Client 2] Received /train request.")
        
        # 2. Train this model on our local data
        new_local_model = train_local_model(global_model, MY_DATA)
        
        # 3. Send back the results to the server
        response = {
            'local_model': new_local_model,
            'data_size': DATA_SIZE  # This is 'n_k'
        }
        return jsonify(response), 200

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

# =============================================================================
# Run the Client App Server
# =============================================================================
if __name__ == "__main__":
    # This client will run on port 5002
    app.run(port=5002, debug=True, host="0.0.0.0")