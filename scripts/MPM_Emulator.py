import os
import subprocess
import csv
import glob
import re

class ObjRenderer:
    def __init__(self, 
                 base_path='results', 
                 GL_render_path="GLRender3d/build/GLRender3d",
                 eyepos=(0.0, 50.0, 0.0),
                 quat=(1.0, 0.0, 0.00, 0.000),
                 window_size=(960, 540),
                 fov=50):
        """
        OBJ renderer with config-specific displacement value passing
        
        Args:
            base_path: Root directory containing simulation results
            GL_render_path: Path to rendering executable
            eyepos: Camera position as (x, y, z)
            quat: Camera rotation quaternion as (w, x, y, z)
            window_size: Render window dimensions as (width, height)
            fov: Field of view angle
        """
        self.base_path = base_path
        self.GL_render_path = GL_render_path
        self.eyepos = eyepos
        self.quat = quat
        self.window_size = window_size
        self.fov = fov
        
        # Prepare camera parameters for rendering command
        self.py_camera_position = f"{eyepos[0]},{eyepos[1]},{eyepos[2]}"
        self.py_camera_quat = f"{quat[0]},{quat[1]},{quat[2]},{quat[3]}"
        self.py_camera_window = f"{window_size[0]},{window_size[1]}"
        self.py_camera_fov = str(fov)
    
    def _generate_render_command(self, obj_path, x_value):
        """Generate rendering command with specific displacement value"""
        out_png = obj_path.replace('.obj', '.png')
        return (f'"{self.GL_render_path}" -a "{out_png}" -b "{obj_path}" '
                f'-c {self.py_camera_position} -d {self.py_camera_quat} '
                f'-e {self.py_camera_fov} -f {self.py_camera_window} '
                f'-g "{x_value}"')
    
    def _get_csv_paths(self):
        """Find all simulation_results.csv files in run directories"""
        run_dirs = glob.glob(os.path.join(self.base_path, 'run_*'))
        csv_paths = []
        for run_dir in run_dirs:
            csv_path = os.path.join(run_dir, "simulation_results.csv")
            if os.path.exists(csv_path):
                csv_paths.append(csv_path)
        return csv_paths
    
    def _get_displacement_value(self, obj_filename, displacements, width):
        """
        Get displacement value based on OBJ filename configuration
        
        Mapping:
        - config_00 -> 0.0
        - config_01 -> x1 (displacement value 1)
        - ...
        - config_08 -> x8 (displacement value 8)
        """
        # Extract configuration number from filename
        match = re.search(r'config_(\d{2})', obj_filename)
        if match:
            config_num = match.group(1)
            if config_num == '00':
                return width
            elif config_num == '08':  # config_08 corresponds to x8
                return f"{float(displacements[7]) + float(width):.2f}" if len(displacements) > 7 else "0.0"
            elif config_num in ['01','02','03','04','05','06','07']:
                # Convert to index: '01'->0, '02'->1, ... '07'->6
                idx = int(config_num) - 1
                return f"{float(displacements[idx]) + float(width):.2f}" if idx < len(displacements) else "0.0"
        return width
    
    def _parse_csv_row(self, row):
        """Parse CSV row and extract parameter information"""
        try:
            n = row[0]
            eta = row[1]
            sigma_y = row[2]
            width = row[3]
            height = row[4]
            displacements = row[5:13]  # x_01 to x_08
            
            # Create sample directory name using parameter values
            sample_dir_name = f"{float(n):.2f}_{float(eta):.2f}_{float(sigma_y):.2f}"
            
            # Create display string for detailed image annotation
            param_str = (f"n={n}, eta={eta}, σ_y={sigma_y}\n"
                         f"w={width}, h={height}\n")
            disp_str = "Displacements:\n" + "\n".join([f"x_{i+1:02d}: {d}" for i, d in enumerate(displacements)])
            
            return sample_dir_name, displacements, param_str + disp_str, width
        except Exception as e:
            print(f"Error parsing CSV row: {e}")
            return None, None, None
    
    def render_all(self):
        """Render all OBJ files with config-specific displacement values"""
        csv_paths = self._get_csv_paths()
        if not csv_paths:
            print(f"No simulation_results.csv files found in {self.base_path}")
            return
        
        processed = 0
        for csv_path in csv_paths:
            run_dir = os.path.dirname(csv_path)
            print(f"Processing run: {os.path.basename(run_dir)}")
            
            try:
                with open(csv_path, 'r') as csvfile:
                    csv_reader = csv.reader(csvfile)
                    next(csv_reader)  # Skip header row
                    
                    for row_idx, row in enumerate(csv_reader):
                        if len(row) < 13:
                            print(f"Skipping invalid row {row_idx}")
                            continue
                        
                        # Get sample directory name from CSV row
                        sample_dir_name, displacements, display_str, width = self._parse_csv_row(row)
                        if not sample_dir_name:
                            continue
                            
                        # Locate sample directory
                        sample_dir = os.path.join(run_dir, sample_dir_name)
                        if not os.path.exists(sample_dir):
                            print(f"Sample directory not found: {sample_dir}")
                            continue
                        
                        # Find all OBJ files in sample directory
                        obj_files = glob.glob(os.path.join(sample_dir, "*.obj"))
                        if not obj_files:
                            print(f"No OBJ files found in {sample_dir}")
                            continue
                        
                        # Render each OBJ file in the directory
                        for obj_path in obj_files:
                            obj_filename = os.path.basename(obj_path)
                            
                            # Find displacement value based on filename config
                            x_value = self._get_displacement_value(obj_filename, displacements, width)

                            print(f"Rendering {obj_filename} with displacement: {x_value}")
                            cmd = self._generate_render_command(obj_path, x_value)
                            subprocess.run(cmd, shell=True)
                            processed += 1
            except Exception as e:
                print(f"Error processing {csv_path}: {str(e)}")
        
        print(f"\nRendering completed! Processed {processed} OBJ files.")

if __name__ == '__main__':
    print("Starting OBJ rendering with config-specific displacement values...")
    renderer = ObjRenderer()
    print(f"Scanning directory: {renderer.base_path}")
    renderer.render_all()
    print("Rendering completed!")
