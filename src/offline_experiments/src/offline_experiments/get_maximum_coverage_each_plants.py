#!/usr/bin/env python3
from nbv_compute.nbv_computor import Nbvcomputor
import rospy
import open3d as o3d
import numpy as np
import torch
import os
import csv

pcd_path =  '/home/jianchao/dataset/nbv_data/paper4/data_smaller_fov_distance'
complete_pcd_dir = '/home/jianchao/dataset/nbv_data/paper4/data_complete'

x_range = np.arange(-20, 20 + 1, 10)
y_range = np.arange(-20, 20 + 1, 10)
z_range = np.arange(0, 20 + 1, 20)
rotation_range = np.arange(0, 360, 90)

row_range = [0, 1, 2, 3, 4, 5, 6, 7, 8] 
col_range = [0, 1, 2, 3, 4, 5, 6, 7, 8]

target_plant = 'PLANT'
test_plants = [3,6,9]
resolution = 0.003
part_pcd_dir = pcd_path
number_runs = 50

# add tested plants to the file name
max_coverage_file = '/home/jianchao/ros_workspace/paper4_semantic_nbv/src/drl_nbv_plan/materials/prilinary_test/max_coverage_%s_%s.csv' % (target_plant, '_'.join(str(plant) for plant in test_plants))
# create a new file for each planner and target
max_coverage_file = open(max_coverage_file, 'a+')
max_coverage_writer = csv.writer(max_coverage_file)

columns = ['plant', 'pose', 'max_coverage']
max_coverage_writer.writerow(columns)


if __name__ == '__main__':
    rospy.init_node('test_part_pcd')
    nbv_compute = Nbvcomputor(octomap_resulotion = resolution, recon_target=target_plant)

    max_coverage_all_plants = []
    for plant in test_plants:
        max_coverage_each_plant = []
        for run in range(number_runs):
            x = np.random.choice(x_range)
            y = np.random.choice(y_range)
            z = np.random.choice(z_range)
            rotation = np.random.choice(rotation_range)

            nbv_compute.load_complete_pc(
                    'plant',
                    plant,
                    plant_rotation=rotation,
                    plant_position=[x, y, z],
                    scale = 1,
                    visual_com = False,
                    visual_roi = False
                )

            plant_path = part_pcd_dir + '/plant%s' % str(plant)+ '/xyzr_%s_%s_%s_%s' % (str(x), str(y), str(z), str(rotation))+ '/pcd'

            init_pcd_filled = False
            for row in row_range:
                for col in col_range:
                    if not init_pcd_filled:
                        acc_pcd = torch.tensor(np.load(plant_path + '/pcd_%s_%s.npy' % (str(row), str(col)))).cuda()
                        acc_pcd = nbv_compute.classify_points_use_bbox(acc_pcd)
                        init_pcd_filled = True
                    else:
                        part_pcd = torch.tensor(np.load(plant_path + '/pcd_%s_%s.npy' % (str(row), str(col)))).cuda()
                        part_pcd = nbv_compute.classify_points_use_bbox(part_pcd)
                        acc_pcd, _, _, _ = nbv_compute.combine_pc(acc_pcd, part_pcd, visual_pc=False, planner='ssl_semantic_nbv')

            _, cr = nbv_compute.cal_coverage(acc_pcd)
            max_coverage_each_plant.append(cr)
            max_coverage_all_plants.append(cr)
            max_coverage_writer.writerow([plant, '%s_%s_%s_%s' % (x, y, z, rotation), cr])
        print('plant', plant, 'max_coverage:', np.array(max_coverage_each_plant).sum()/len(max_coverage_each_plant))
    print('test_plants', test_plants, 'max_coverage:', np.array(max_coverage_all_plants).sum()/len(max_coverage_all_plants))




    
            
    

    


    




