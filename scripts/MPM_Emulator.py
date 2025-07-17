import os
import numpy as np
import pandas as pd
import subprocess

# ti.init(arch=ti.gpu)

def load_x_distance(file_path = 'x_diff_data.csv'):
    data = pd.read_csv(file_path)
    x = 0.0
    return x 

class ObjRenderer:
    def __init__(self, base_path='data', 
                 GL_render_path="GLRender3d/build/GLRender3d",
                 eyepos=np.array([0.0, 50.0, 0.0]),
                 quat=np.array([-1.0, 0.0, 0.00, 0.000]),
                 window_size=np.array([960, 540]),
                 fov=36.17541,
                 x_diff_file='x_diff_data.csv'):
        """
        Initialize renderer parameters
        
        Args:
            base_path: Root directory containing model folders
            GL_render_path: Path to renderer executable
            eyepos: Camera position [x, y, z]
            quat: Camera rotation quaternion [w, x, y, z]
            window_size: Render window dimensions [width, height]
            fov: Field of view angle
            x_diff_file: CSV file containing x_diff data
        """
        self.base_path = base_path
        self.GL_render_path = GL_render_path
        self.eyepos = eyepos
        self.quat = quat
        self.window_size = window_size
        self.fov = fov
        self.x_diff_file = x_diff_file
        
        # Prepare renderer command parameters
        self.py_camera_position = f"{eyepos[0]},{eyepos[1]},{eyepos[2]}"
        self.py_camera_quat = f"{quat[0]},{quat[1]},{quat[2]},{quat[3]}"
        self.py_camera_window = f"{window_size[0]},{window_size[1]}"
        self.py_camera_fov = str(fov)
        
        # Load x_diff data
        self.x_diff_data = pd.read_csv(x_diff_file)
    
    def _generate_render_command(self, obj_path, x_distance):
        """
        Generate render command string
        
        Args:
            obj_path: Path to .obj file
            x_distance: Render distance parameter
            
        Returns:
            str: Complete render command
        """
        out_png = obj_path.replace('.obj', '.png')
        return (f'"{self.GL_render_path}" -a "{out_png}" -b "{obj_path}" '
                f'-c {self.py_camera_position} -d {self.py_camera_quat} '
                f'-e {self.py_camera_fov} -f {self.py_camera_window} '
                f'-g {x_distance}')
    
    def render_all(self):
        """
        Render all qualifying .obj files in base_path directory
        """
        # Iterate through all parameter folders
        for parameter_folder in os.listdir(self.base_path):
            folder_path = os.path.join(self.base_path, parameter_folder)
            if not os.path.isdir(folder_path):
                continue
                
            # Process each setup in parameter folder
            for setup in os.listdir(folder_path):
                setup_path = os.path.join(folder_path, setup)
                if not os.path.isdir(setup_path):
                    continue
                    
                # Parse setup parameters from folder name
                parts = setup.split('_')
                if len(parts) < 3:
                    continue
                    
                try:
                    setup_01 = float(parts[1])
                    setup_02 = float(parts[2])
                except ValueError:
                    continue
                
                # Handle special config file (config_00)
                config_00_path = os.path.join(setup_path, 'config_00.obj')
                if os.path.exists(config_00_path):
                    cmd = self._generate_render_command(config_00_path, str(setup_02))
                    print(f"Rendering: {config_00_path}")
                    subprocess.run(cmd, shell=True)
                
                # Process data-related config files
                for _, row in self.x_diff_data.iterrows():
                    # Check both sets of parameters in CSV
                    for prefix in [('x_01', 'x_02'), ('x_11', 'x_12')]:
                        col1, col2 = prefix
                        if setup_01 == row[col1] and setup_02 == row[col2]:
                            # Determine starting index based on parameter set
                            start_idx = 3 if prefix == ('x_01', 'x_02') else 13
                            
                            # Process 8 config files
                            for j in range(8):
                                config_num = j + 1
                                config_file = f'config_{config_num:02d}.obj'
                                obj_path = os.path.join(setup_path, config_file)
                                
                                if not os.path.exists(obj_path):
                                    continue
                                    
                                # Calculate distance and render
                                col_name = f'x_{start_idx + j:02d}'
                                x_distance = str(row[col_name] + setup_02)
                                cmd = self._generate_render_command(obj_path, x_distance)
                                print(f"Rendering: {obj_path}")
                                subprocess.run(cmd, shell=True)

# Maintain original script functionality
if __name__ == '__main__':
    # Create default renderer and execute
    renderer = ObjRenderer()
    renderer.render_all()