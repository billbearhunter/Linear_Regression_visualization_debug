import os
import numpy as np
import subprocess

class ObjRenderer:
    def __init__(self, 
                 base_path='results/run_20250718_133749', 
                 GL_render_path="GLRender3d/build/GLRender3d",
                 eyepos=np.array([0.0, 50.0, 0.0]),
                 quat=np.array([-1.0, 0.0, 0.00, 0.000]),
                 window_size=np.array([960, 540]),
                 fov=36.17541):
        """
        Simplified OBJ renderer: Renders all OBJ files to PNG
        
        Args:
            base_path: Root directory containing model folders
            GL_render_path: Path to renderer executable
            eyepos: Camera position [x, y, z]
            quat: Camera rotation quaternion [w, x, y, z]
            window_size: Render window dimensions [width, height]
            fov: Field of view angle
        """
        self.base_path = base_path
        self.GL_render_path = GL_render_path
        self.eyepos = eyepos
        self.quat = quat
        self.window_size = window_size
        self.fov = fov
        
        # Prepare renderer command parameters
        self.py_camera_position = f"{eyepos[0]},{eyepos[1]},{eyepos[2]}"
        self.py_camera_quat = f"{quat[0]},{quat[1]},{quat[2]},{quat[3]}"
        self.py_camera_window = f"{window_size[0]},{window_size[1]}"
        self.py_camera_fov = str(fov)
    
    def _generate_render_command(self, obj_path):
        """
        Generate render command string
        
        Args:
            obj_path: Path to .obj file
            
        Returns:
            str: Complete render command
        """
        out_png = obj_path.replace('.obj', '.png')
        return (f'"{self.GL_render_path}" -a "{out_png}" -b "{obj_path}" '
                f'-c {self.py_camera_position} -d {self.py_camera_quat} '
                f'-e {self.py_camera_fov} -f {self.py_camera_window} '
                '-g 0.0')  # Fixed x_distance value
    
    def render_all(self):
        """
        Render all OBJ files found in base_path subdirectories
        """
        # Recursively scan all directories
        for root, _, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.obj'):
                    obj_path = os.path.join(root, file)
                    cmd = self._generate_render_command(obj_path)
                    print(cmd)
                    print(f"Rendering: {obj_path}")
                    subprocess.run(cmd, shell=True)

# Script entry point
if __name__ == '__main__':
    print("Starting OBJ rendering process...")
    renderer = ObjRenderer()
    print(f"Scanning directory: {renderer.base_path}")
    renderer.render_all()
    print("Rendering completed!")
