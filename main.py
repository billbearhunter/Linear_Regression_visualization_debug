import os
import json
import random
import numpy as np
import traceback
import taichi as ti
import joblib as job
from datetime import datetime
from config.config import (
    MIN_ETA, MAX_ETA,
    MIN_N, MAX_N,
    MIN_SIGMA_Y, MAX_SIGMA_Y,
    MIN_WIDTH, MAX_WIDTH,
    MIN_HEIGHT, MAX_HEIGHT,
    XML_TEMPLATE_PATH
)

from simulation.taichi import MPMSimulator
from scripts import MPM_Emulator

ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)

sample_dir = "results/samples"

# 1. Load pretrained prediction model
def load_prediction_model(model_path):
    """Load pretrained flow distance prediction model"""
    try:
        model = job.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {str(e)}")
        raise e

# 2. Generate random parameter sets
def generate_random_parameters(num_samples=1):
    """Generate random parameter samples within defined valid ranges"""
    params = []
    for _ in range(num_samples):
        params.append({
            'n': random.uniform(MIN_N, MAX_N),
            'eta': random.uniform(MIN_ETA, MAX_ETA),
            'sigma_y': random.uniform(MIN_SIGMA_Y, MAX_SIGMA_Y),
            'width': random.uniform(MIN_WIDTH, MAX_WIDTH),
            'height': random.uniform(MIN_HEIGHT, MAX_HEIGHT)
        })
    return params


def main():
    """Main workflow execution function"""
    # Initialize
    print("Starting fluid simulation prediction workflow")
    print("=" * 60)
    random.seed(66)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join("results", f"run_{timestamp}")
    
    # # Load prediction model
    print("Loading prediction model...")
    model = load_prediction_model("model/best_model.joblib")
    
    # Generate parameter sets
    print("Generating parameter configurations...")
    params_list = generate_random_parameters(1)
    
    # Track results
    all_results = {}
    print("Generated parameter sets:")
    
    for i, params in enumerate(params_list):
        sample_id = f"sample_{i+1}"
        print(f"\n{'=' * 40}")
        print(f"Processing sample #{i+1}: {sample_id}")
        print("=" * 40)
        print(f"Parameters: n={params['n']:.4f}, eta={params['eta']:.2f}, "
              f"σ_y={params['sigma_y']:.2f}, width={params['width']:.2f}, height={params['height']:.2f}")
        

        
        # Save parameters and use model to predict flow distance
        try:
            # Initialize MPM simulator with parameters
            # Initialize simulator
            simulator = MPMSimulator(XML_TEMPLATE_PATH)
            # Run simulation
            simulator.configure_geometry(width=params['width'], height=params['height'])
            displacements = simulator.run_simulation(n=params['n'],
                                                    eta=params['eta'],
                                                    sigma_y=params['sigma_y'],
                                                    output_dir=results_root)
            print(f"Simulation completed for sample {sample_id}. Displacements: {displacements}")
            # Predict flow distance using the model
            prediction = model.predict([[params['n'], params['eta'], params['sigma_y'], 
                                            params['width'], params['height']]])[0]
            print(f"Predicted flow distance: {prediction}")
            flow_distance = prediction.tolist()
            # Save results
            result = {
                'params': params,
                'predicted_flow_distance': flow_distance,
                # 'simulation_data': simulator.get_simulation_data()  # Assuming this method exists
            }
            all_results[sample_id] = result
            result_file = os.path.join(results_root, f"{sample_id}_result.json")
            os.makedirs(os.path.dirname(result_file), exist_ok=True)
            with open(result_file, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"Results saved to {result_file}")
        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}") 
            traceback.print_exc()
        print(f"Sample {sample_id} processing complete.\n")


if __name__ == "__main__":
    main()