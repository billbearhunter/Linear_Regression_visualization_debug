import os
import json
import random
import torch
import numpy as np
import traceback
import taichi as ti
from datetime import datetime
from config.config import (
    MIN_ETA, MAX_ETA,
    MIN_N, MAX_N,
    MIN_SIGMA_Y, MAX_SIGMA_Y,
    MIN_WIDTH, MAX_WIDTH,
    MIN_HEIGHT, MAX_HEIGHT
)

from simulation.taichi import MPMSimulator
from simulation.xmlParser import MPMXMLData
# from rendering.render_utils import generate_objs_and_render
# from src.utils.visualization import visualize_comparison
# from src.utils.file_ops import ensure_directory

ti.init(arch=ti.gpu, offline_cache=True, default_fp=ti.f32, default_ip=ti.i32)

# 1. Load pretrained prediction model
def load_prediction_model(model_path):
    """Load pretrained flow distance prediction model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        model = torch.load(model_path, map_location=device)
        model.eval()
        print("Prediction model loaded successfully")
        return model
    except Exception as e:
        print(f"Model loading failed: {str(e)}")
        raise

# 2. Generate random parameter sets
def generate_random_parameters(num_samples=5):
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
    random.seed(42)
    torch.manual_seed(42)
    
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_root = os.path.join("results", f"run_{timestamp}")
    
    # Load prediction model
    print("Loading prediction model...")
    model = load_prediction_model("model/best_model.joblib")
    
    # Generate parameter sets
    print("Generating parameter configurations...")
    params_list = generate_random_parameters(5)
    
    # Track results
    all_results = {}
    
    for i, params in enumerate(params_list):
        sample_id = f"sample_{i+1}"
        print(f"\n{'=' * 40}")
        print(f"Processing sample #{i+1}: {sample_id}")
        print("=" * 40)
        print(f"Parameters: n={params['n']:.4f}, eta={params['eta']:.2f}, "
              f"σ_y={params['sigma_y']:.2f}, width={params['width']:.2f}, height={params['height']:.2f}")
        

        
        # # Save parameters
        # with open(os.path.join(sample_dir, "parameters.json"), "w") as file:
        #     json.dump(params, file, indent=4)
        
        # try:
        #     # Predict flow distances
        #     input_tensor = torch.tensor([
        #         params['n'], 
        #         params['eta'], 
        #         params['sigma_y'], 
        #         params['width'], 
        #         params['height']
        #     ], dtype=torch.float32).unsqueeze(0)
            
        #     with torch.no_grad():
        #         predicted_flow = model(input_tensor).numpy()[0]
            
        #     print(f"Predicted flow distances: {predicted_flow}")

            
        # except Exception as e:
        #     print(f"Error processing sample #{i+1}: {str(e)}")
        #     traceback.print_exc()
        #     with open(os.path.join(sample_dir, "error.log"), "w") as file:
        #         file.write(f"Sample processing failed\nError: {str(e)}\n")
        #         file.write(traceback.format_exc())
    

if __name__ == "__main__":
    main()