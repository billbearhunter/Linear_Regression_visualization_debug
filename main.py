import os
import json
import csv
import random
import numpy as np
import traceback
import taichi as ti
import joblib as job
from datetime import datetime
from sklearn.preprocessing import StandardScaler
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
def load_prediction_model(model_path, y_scaler_path):
    """Load pretrained flow distance prediction model"""
    try:
        model = job.load(model_path)
        y_scaler = job.load(y_scaler_path)
        # feature_scaler = job.load(feature_scaler_path)
        print(f"Model loaded successfully from {model_path}")
        return model, y_scaler
    except Exception as e:
        print(f"Failed to load model from {model_path}: {str(e)}")
        raise e

# 2. Generate random parameter sets
def generate_random_parameters(num_samples=2):
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
    model, y_scaler = load_prediction_model("model/best_model.joblib", "model/target_scaler.joblib")

    # Generate parameter sets
    print("Generating parameter configurations...")
    params_list = generate_random_parameters(5)
    
    csv_filename = os.path.join(results_root, "simulation_results.csv")
    os.makedirs(results_root, exist_ok=True)
    with open(csv_filename, 'w') as f:
        headers = ["n", "eta", "sigma_y", "width", "height"] + [f"x_{0}{i+1}" for i in range(8)]
        f.write(",".join(headers) + "\n")

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
        
        sample_dir_path = os.path.join(results_root, f"{params['n']:.2f}_{params['eta']:.2f}_{params['sigma_y']:.2f}")
        os.makedirs(sample_dir_path, exist_ok=True)
        
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
                                                    output_dir= sample_dir_path)
            print(f"Simulation completed for sample {sample_id}. Displacements: {displacements}")
            # Predict flow distance using the model
            prediction = model.predict([[params['n'], params['eta'], params['sigma_y'], 
                                           params['width'], params['height']]])[0]
            prediction_scaled = y_scaler.inverse_transform(prediction.reshape(1, -1))
            print(f"Predicted flow distance: {prediction_scaled[0]}")
            flow_distance = prediction_scaled[0].tolist()
            # Append results to CSV file
            with open(csv_filename, 'a', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                # Write data row
                csv_writer.writerow([
                    params['n'],
                    params['eta'],
                    params['sigma_y'],
                    params['width'],
                    params['height'],
                    # *[f"{d:.15f}" for d in flow_distance]
                    *[displacements[i] if i < len(displacements) else 0 for i in range(8)]
                ])
            
            # # Save detailed results to JSON
            # result = {
            #     'params': params,
            #     'predicted_flow_distance': prediction.tolist(),
            #     'displacements': displacements,
            # }
            # all_results[sample_id] = result
            # result_file = os.path.join(results_root, f"{sample_id}_result.json")
            # with open(result_file, 'w') as f:
            #     json.dump(result, f, indent=4)
            # print(f"Results saved to {result_file}")

 

        except Exception as e:
            print(f"Error processing sample {sample_id}: {str(e)}") 
            traceback.print_exc()
        print(f"Sample {sample_id} processing complete.\n")

    # Render OBJ files using MPM_Emulator
    print("Rendering OBJ files...")
    renderer = MPM_Emulator.MPMEmulator()
    renderer.render_all()  


if __name__ == "__main__":
    main()