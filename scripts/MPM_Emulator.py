import os
import sys
import numpy as np
import taichi as ti
import pandas as pd
import subprocess

ti.init(arch=ti.gpu)

def load_x_distance(file_path = 'x_diff_data.csv'):
    data = pd.read_csv(file_path)
    x = 0.0
    return x 


def main():
    base_path = 'data'
    GL_render_path="GLRender3d/build/GLRender3d" 
    # eyepos = np.array([8.45696138, 14.32447243, 25.97375739])
    # quat = np.array([0.96642661, -0.25693407, 0.00158011, -0.00141509])
    eyepos = np.array([0.0, 50.0, 0.0])
    quat = np.array([-1.0, 0.0, 0.00, 0.000])
    window_size = np.array([960, 540])
    fov = 36.17541
    py_camera_position = str(eyepos[0]) + "," + str(eyepos[1])  + "," + str(eyepos[2])
    py_camera_quat = str(quat[0]) + "," + str(quat[1]) + "," + str(quat[2]) + "," +str(quat[3])
    py_camera_window = str(window_size[0]) + "," + str(window_size[1])
    py_camera_fov = str(fov)
    x_distance = str(0.0)

    file_path = 'x_diff_data.csv'
    data = pd.read_csv(file_path)
    # print(data)

    for parameter_folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, parameter_folder)
        # n = float(parameter_folder.split('_')[1])
        # eta = float(parameter_folder.split('_')[3])
        # sigma_y = float(parameter_folder.split('_')[6])
        # print(folder_path)
        if os.path.isdir(folder_path):
            for setup in os.listdir(folder_path):
                setup_path = os.path.join(folder_path, setup)
                if os.path.isdir(setup_path):
                    setup_01 = float(setup.split('_')[1])
                    setup_02 = float(setup.split('_')[2])

                    for i in range(len(data)):
                        if setup_01 == data[f'x_01'][i] and setup_02 == data[f'x_02'][i]:
                            for j in range(8):
                                x_distance = str(data[f'x_{j+3:02d}'][i] + setup_02)
                                # print(f'x_{j+3:02d} = {x_distance}')  
                                obj_file_path = os.path.join(setup_path, f'config_0{j+1}.obj')
                                # print(obj_file_path)
                                outpngFilePath = obj_file_path.replace('.obj', '.png')
                                # print(outpngFilePath)
                                cmd = '"' + GL_render_path + '" -a "' + outpngFilePath + '" -b "' + obj_file_path + '" -c ' + py_camera_position + ' -d ' + py_camera_quat + ' -e ' + py_camera_fov + ' -f ' + py_camera_window + ' -g ' + x_distance
                                #cmd = '"' + GL_render_path + '" -a "' + outpngFilePath + '" -b "' + obj_file_path + '" -c ' + py_camera_position + ' -d ' + py_camera_quat + ' -e ' + py_camera_fov + ' -f ' + py_camera_window
                                print(cmd)
                                subprocess.run(cmd, shell=True)

                        elif setup_01 == data[f'x_11'][i] and setup_02 == data[f'x_12'][i]:
                            for j in range(8):
                                x_distance = str(data[f'x_{j+13:02d}'][i] + setup_02)
                                # print(f'x_{j+13:02d} = {x_distance}')
                                obj_file_path = os.path.join(setup_path, f'config_0{j+1}.obj')
                                # print(obj_file_path)
                                outpngFilePath = obj_file_path.replace('.obj', '.png')
                                # print(outpngFilePath)
                                cmd = '"' + GL_render_path + '" -a "' + outpngFilePath + '" -b "' + obj_file_path + '" -c ' + py_camera_position + ' -d ' + py_camera_quat + ' -e ' + py_camera_fov + ' -f ' + py_camera_window + ' -g ' + x_distance
                                #cmd = '"' + GL_render_path + '" -a "' + outpngFilePath + '" -b "' + obj_file_path + '" -c ' + py_camera_position + ' -d ' + py_camera_quat + ' -e ' + py_camera_fov + ' -f ' + py_camera_window
                                print(cmd)
                                subprocess.run(cmd, shell=True)

                    obj_file_path = os.path.join(setup_path, f'config_00.obj')
                    outpngFilePath = obj_file_path.replace('.obj', '.png')
                    x_distance = str(setup_02)
                    cmd = '"' + GL_render_path + '" -a "' + outpngFilePath + '" -b "' + obj_file_path + '" -c ' + py_camera_position + ' -d ' + py_camera_quat + ' -e ' + py_camera_fov + ' -f ' + py_camera_window + ' -g ' + x_distance
                    #cmd = '"' + GL_render_path + '" -a "' + outpngFilePath + '" -b "' + obj_file_path + '" -c ' + py_camera_position + ' -d ' + py_camera_quat + ' -e ' + py_camera_fov + ' -f ' + py_camera_window
                    print(cmd)
                    subprocess.run(cmd, shell=True)
                    
            
if __name__ == '__main__':
    main()