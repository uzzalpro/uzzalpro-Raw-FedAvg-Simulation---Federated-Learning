import numpy as np
import requests
from flask import Flask, jsonify
import time # To simulate time passing between rounds

# --- Hyperparameters ---
GLOBAL_ROUNDS = 5 # t: Number of times we aggregate
CLIENT_URLS = [
    {'id': 1, 'url': 'http://127.0.0.1:5001/train'},
    {'id': 2, 'url': 'http://127.0.0.1:5002/train'}
]

# --- Global Model State ---
# The server starts with a "dumb" model (w=0, b=0)
GLOBAL_WEIGHTS = {'w': 0.0, 'b': 0.0}

app = Flask(__name__)

# =============================================================================
# Core FedAvg Logic: The Aggregation Math
# =============================================================================
def aggregate_models(client_updates):
    """
    Performs the Federated Averaging (FedAvg) aggregation.
    Math: w_t+1 = sum( (n_k / N) * w_k )
    """
    
    # 1. Collect all data sizes (n_k) and calculate Total Data Size (N)
    client_data_sizes = [update['data_size'] for update in client_updates]
    total_data_size = sum(client_data_sizes) # This is N
    
    # Initialize new global weights to zero
    new_w = 0.0
    new_b = 0.0
    
    print("\n--- FedAvg AGGREGATION MATH ---")
    print(f"Total Data Points (N): {total_data_size}")

    for i, update in enumerate(client_updates):
        client_id = CLIENT_URLS[i]['id']
        local_model = update['local_model']  # w_k, b_k
        n_k = update['data_size']          # n_k
        
        # Calculate the client's aggregation weight
        # FedAvg Weight = n_k / N
        aggregation_weight = n_k / total_data_size 
        
        # --- VISUAL MATH STEP ---
        print(f"\n[Client {client_id}] Data (n_k): {n_k}")
        print(f"  Aggregation Weight (n_k/N): {aggregation_weight:.4f}")
        print(f"  Local Model: w={local_model['w']:.4f}, b={local_model['b']:.4f}")
        
        # Apply the weighted average
        weighted_w = aggregation_weight * local_model['w']
        weighted_b = aggregation_weight * local_model['b']
        
        print(f"  Weighted Update: w_weighted={weighted_w:.4f}, b_weighted={weighted_b:.4f}")

        # Sum the weighted updates
        new_w += weighted_w
        new_b += weighted_b
        
    print("\n--- FINAL GLOBAL MODEL CALCULATION ---")
    print(f"New Global W: {new_w:.4f}")
    print(f"New Global B: {new_b:.4f}")
    
    return {'w': new_w, 'b': new_b}


# =============================================================================
# The Main Orchestration Logic
# =============================================================================
def run_global_round(global_round):
    global GLOBAL_WEIGHTS
    print(f"\n=======================================================")
    print(f"|           STARTING GLOBAL ROUND {global_round} / {GLOBAL_ROUNDS}           |")
    print(f"| Global Model to Distribute: w={GLOBAL_WEIGHTS['w']:.4f}, b={GLOBAL_WEIGHTS['b']:.4f} |")
    print(f"=======================================================")
    
    # 1. DISTRIBUTION: Send model to all clients
    client_updates = []
    
    for client in CLIENT_URLS:
        try:
            print(f"  -> Sending model to Client {client['id']} at {client['url']}...")
            
            # Use requests to make a POST call to the client's /train endpoint
            response = requests.post(
                client['url'],
                json={'global_model': GLOBAL_WEIGHTS},
                timeout=10 # Set a timeout
            )
            response.raise_for_status() # Raise an error for bad status codes
            
            # 2. COLLECTION: Receive the trained local model and data size (n_k)
            client_updates.append(response.json())
            
            print(f"  <- Received update from Client {client['id']} successfully.")

        except requests.exceptions.RequestException as e:
            # Handle clients that are offline or fail
            print(f"  !!! ERROR: Could not reach Client {client['id']}. Skipping this client: {e}")
    
    if not client_updates:
        print("No clients responded. Stopping aggregation for this round.")
        return

    # 3. AGGREGATION: Apply the FedAvg math
    new_global_weights = aggregate_models(client_updates)
    
    # 4. UPDATE: Save the new global model for the next round
    GLOBAL_WEIGHTS = new_global_weights
    
    print(f"\n>>> ROUND {global_round} COMPLETE. New Global Model: w={GLOBAL_WEIGHTS['w']:.4f}, b={GLOBAL_WEIGHTS['b']:.4f}")

# =============================================================================
# API Endpoints
# =============================================================================

@app.route('/start', methods=['POST'])
def start_federated_learning():
    """Starts the global FedAvg process."""
    print("-----------------------------------------")
    print("        FEDERATED LEARNING STARTED       ")
    print("-----------------------------------------")
    
    global GLOBAL_WEIGHTS
    
    # Reset model to initial state
    GLOBAL_WEIGHTS = {'w': 0.0, 'b': 0.0}

    for t in range(1, GLOBAL_ROUNDS + 1):
        run_global_round(t)
        # Simulate time passing before the next round
        time.sleep(1) 

    return jsonify({
        'message': 'Federated Learning simulation finished.',
        'final_model': GLOBAL_WEIGHTS
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Endpoint to check current model weights."""
    return jsonify({
        'current_global_model': GLOBAL_WEIGHTS,
        'global_rounds_planned': GLOBAL_ROUNDS
    })

if __name__ == "__main__":
    # The server runs on port 5000
    app.run(port=5000, debug=True, host="0.0.0.0")