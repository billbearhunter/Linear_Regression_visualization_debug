import os
import json
import random
import torch
import numpy as np
import traceback
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

# 3. Update XML configuration with parameters
def update_xml_config(template_path, output_path, params):
    """Update XML configuration file with specified parameters"""
    try:
        with open(template_path, 'r') as file:
            content = file.read()
        
        # Update material parameters
        content = content.replace('herschel_bulkley_power="1.0"', 
                                f'herschel_bulkley_power="{params["n"]}"')
        content = content.replace('eta="300.0"', 
                                f'eta="{params["eta"]}"')
        content = content.replace('yield_stress="400.0"', 
                                f'yield_stress="{params["sigma_y"]}"')
        
        # Update geometry dimensions
        content = content.replace('max="7.0 7.0 4.1500000"', 
                                f'max="{params["width"]} {params["width"]} {params["height"]}"')
        
        # Update static box positions
        content = content.replace(
            '<static_box min="-1.000000 0.000000 -0.300000" max="7.0 20.000000 0.000000"',
            f'<static_box min="-1.000000 0.000000 -0.300000" max="{params["width"]} 20.000000 0.000000"'
        )
        content = content.replace(
            '<static_box min="-1.000000 0.000000 4.000000" max="7.0 20.000000 4.300000"',
            f'<static_box min="-1.000000 0.000000 4.000000" max="{params["width"]} 20.000000 4.300000"'
        )
        
        with open(output_path, 'w') as file:
            file.write(content)
        
        print(f"XML configuration updated: {output_path}")
        return True
    except Exception as e:
        print(f"XML configuration update failed: {str(e)}")
        return False

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
    ensure_directory(results_root)
    
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
        print(f"🧪 Processing sample #{i+1}: {sample_id}")
        print("=" * 40)
        print(f"Parameters: n={params['n']:.4f}, eta={params['eta']:.2f}, "
              f"σ_y={params['sigma_y']:.2f}, width={params['width']:.2f}, height={params['height']:.2f}")
        
        # Create sample directory
        sample_dir = os.path.join(results_root, sample_id)
        ensure_directory(sample_dir)
        
        # Save parameters
        with open(os.path.join(sample_dir, "parameters.json"), "w") as file:
            json.dump(params, file, indent=4)
        
        try:
            # Predict flow distances
            input_tensor = torch.tensor([
                params['n'], 
                params['eta'], 
                params['sigma_y'], 
                params['width'], 
                params['height']
            ], dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                predicted_flow = model(input_tensor).numpy()[0]
            
            print(f"📊 Predicted flow distances: {predicted_flow}")
            
            # Update XML configuration
            config_path = os.path.join(sample_dir, "config.xml")
            if not update_xml_config("config/setting.xml", config_path, params):
                raise RuntimeError("XML configuration failed")
            
            # Run MPM simulation
            print("⚙️ Starting MPM simulation...")
            xml_config = MPMXMLData(config_path)
            simulator = MPMSimulator(xml_config)
            
            if not simulator.run_simulation():
                raise RuntimeError("MPM simulation failed")
            
            # Get actual flow distances
            actual_flow = simulator.calculate_flow_distances()
            print(f"📏 Actual flow distances: {actual_flow}")
            
            # Generate OBJs and renders
            print("🎨 Generating visualization assets...")
            generate_objs_and_render(
                input_dir=simulator.output_dir,
                sample_dir=sample_dir
            )
            
            # Create annotated comparison images
            print("🖌️ Generating comparison visualizations...")
            visualize_comparison(
                render_dir=os.path.join(sample_dir, "renders"),
                output_dir=sample_dir,
                predicted_flow=predicted_flow,
                actual_flow=actual_flow
            )
            
            # Save results
            sample_results = {
                "parameters": params,
                "predicted_flow": predicted_flow.tolist(),
                "actual_flow": actual_flow.tolist()
            }
            with open(os.path.join(sample_dir, "results.json"), "w") as file:
                json.dump(sample_results, file, indent=4)
            
            all_results[sample_id] = sample_results
            print(f"✅ Sample #{i+1} processed successfully! Results in {sample_dir}")
            
        except Exception as e:
            print(f"⚠️ Error processing sample #{i+1}: {str(e)}")
            traceback.print_exc()
            with open(os.path.join(sample_dir, "error.log"), "w") as file:
                file.write(f"Sample processing failed\nError: {str(e)}\n")
                file.write(traceback.format_exc())
    
    # Generate summary report
    print("\n📈 Generating summary report...")
    generate_summary_report(all_results, os.path.join(results_root, "summary_report.html"))
    
    print(f"\n🎉 Workflow completed! All results in {results_root}")

def generate_summary_report(results, report_path):
    """Generate HTML summary report of all results"""
    try:
        with open(report_path, "w") as file:
            file.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <title>Fluid Simulation Prediction Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #2c3e50; }
                    table { border-collapse: collapse; width: 100%; margin-bottom: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #f2f2f2; }
                    .comparison-img { max-width: 200px; height: auto; }
                </style>
            </head>
            <body>
                <h1>Fluid Simulation Prediction Report</h1>
                <p>Generated on: {timestamp}</p>
            """.format(timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            
            # Summary table
            file.write("<h2>Parameter Summary</h2>")
            file.write("<table>")
            file.write("<tr><th>Sample</th><th>n</th><th>η</th><th>σ<sub>y</sub></th><th>Width</th><th>Height</th></tr>")
            
            for sample_id, data in results.items():
                p = data["parameters"]
                file.write(f"<tr><td>{sample_id}</td><td>{p['n']:.4f}</td><td>{p['eta']:.2f}</td>")
                file.write(f"<td>{p['sigma_y']:.2f}</td><td>{p['width']:.2f}</td><td>{p['height']:.2f}</td></tr>")
            
            file.write("</table>")
            
            # Detailed results
            file.write("<h2>Detailed Results</h2>")
            for sample_id, data in results.items():
                file.write(f"<h3>{sample_id}</h3>")
                file.write(f"<p>Sample directory: results/{sample_id}</p>")
                
                # Flow distance comparison
                file.write("<table>")
                file.write("<tr><th>Frame</th><th>Predicted</th><th>Actual</th><th>Difference</th></tr>")
                pred = data["predicted_flow"]
                actual = data["actual_flow"]
                
                for frame in range(8):
                    diff = abs(pred[frame] - actual[frame])
                    file.write(f"<tr><td>{frame+1}</td><td>{pred[frame]:.4f}</td>")
                    file.write(f"<td>{actual[frame]:.4f}</td><td>{diff:.4f}</td></tr>")
                
                file.write("</table>")
                
                # Image comparison (first frame)
                img_path = f"{sample_id}/comparison_00.png"
                file.write(f"<img src='{img_path}' alt='Comparison visualization' class='comparison-img'>")
            
            file.write("</body></html>")
        
        print(f"Report generated: {report_path}")
        return True
    except Exception as e:
        print(f"Report generation failed: {str(e)}")
        return False

if __name__ == "__main__":
    main()