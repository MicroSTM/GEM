# generate a pkl file with all the task info

import numpy as np
import pickle
import os
import pdb
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--task-dir", type=str)  
args = parser.parse_args()

if not os.path.isdir(args.task_dir):
    os.makedirs(args.task_dir)

#info: 
# G
# predicates (0/1 distractors)
PREDICATES = ["right", "above", "close", "dist", "orien"]  # left, below symmetric
# ('pred', obj1, obj2, x for 'dist)
# LMA instructions for expert
# #change these at test
# init pos
# init shape
# init color
# num agents (test seen at train)
# init size??

n_train_scenarios = 6
n_test_scenarios = 1  # 5

all_predicates = [[[('right',0,1)], #0
                   [('above',1,2)], #1
                   [('right',0,1),('close',0,2)], #2
                   [('above',1,0),('dist',1,2,4.5)], #3
                   [('right',2,1),('above',0,2),('close',1,2)], #4
                   [('close',0,1)], #5
                   [('dist_gt',0,1,6)]], #6
                   #3 obj
                  [[('dist',0,1,5.5),('right',3,2)], #0
                   [('diag',0,1),('dist_gt',2,0,5)], #1
                   [('above',2,3),('close',2,3)]], #2
                   #4 obj
                #   [[('right',0,4)], [('close',4,3)], [('above',1,2),('dist',0,3,7)], [('close',1,4),('close',2,3)], [('right',0,2),('dist',4,2,9),('close',3,2)]], #5 obj
                  ] #choose 1/2/3                       
all_expert_subgoals = [[[['LMA',0,(18,12),1]], #0
                        [['LMA',1,(18,14),1]], #1
                        [['LMA',2,(24,5),1]], #2
                        [['LMA',1,(14,16),1],['LMA',2,(12,12),1]], #3
                        [['LMA',2,(9,4),1],['LMA',0,(9,16),1]], #4
                        [['LMA',0,(15,9),1]], #5
                        [['LMA',0,(14,11),1]]], #6
                        #3 obj
                      [[['LMA',1,(12,16),1], ['LMA',3,(20,4),1]], #0
                       [['LMA',2,(18,15),1], ['LMA',0,(16,9),1]], #1
                       [['LMA',2,(18,18),1]]], #2
                       #4 obj
                #   [[('right',0,4)], [('close',4,3)], [('above',1,2),('dist',0,3,7)], [('close',1,4),('close',2,3)], [('right',0,2),('dist',4,2,9),('close',3,2)]], #5 obj
                  ]
#
train_init_pos = [
    [[16, 18, 13], #0
     [10, 5, 13], #1
     [7, 17, 6], #2
     [11, 12, 2], #3
     [10, 3, 13], #4
     [18, 11, 9], #5
     [11, 15, 10]], #6
     #3 obj
    [[0, 10, 17, 13], #0
     [15, 11, 9, 10], #1
     [16, 4, 8, 12]], #2
     #4 obj
    #   [[('right',0,4)], [('close',4,3)], [('above',1,2),('dist',0,3,7)], [('close',1,4),('close',2,3)], [('right',0,2),('dist',4,2,9),('close',3,2)]], #5 obj
]
train_shapes = [
    [[0, 0, 1], #0
     [0, 1, 2], #1
     [4, 1, 3], #2
     [2, 3, 3], #3
     [2, 0, 4], #4
     [0, 0, 1], #5
     [3, 1, 0]], #6
     #3 obj
    [[2,3,2,0], #0
     [3,2,4,1], #1
     [1,0,0,0]], #2
    #4 obj
    #   [[('right',0,4)], [('close',4,3)], [('above',1,2),('dist',0,3,7)], [('close',1,4),('close',2,3)], [('right',0,2),('dist',4,2,9),('close',3,2)]], #5 obj
]
train_colors = [
    [[1, 2, 3], #0
     [4, 0, 1], #1
     [0, 0, 2], #2
     [2, 3, 1], #3
     [4, 1, 3], #4
     [1, 2, 3],  #5
     [0, 1, 2]], #6
     #3 obj
    [[3,2,1,4], #0
     [0,1,2,0], #1
     [4,3,0,1]], #2
    #4 obj     
    #   [[('right',0,4)], [('close',4,3)], [('above',1,2),('dist',0,3,7)], [('close',1,4),('close',2,3)], [('right',0,2),('dist',4,2,9),('close',3,2)]], #5 obj
]
#
test_init_pos = [[[[2,10,3]], [[5,8,1]], [[13,6,17]], [[1,11,16]], [[10,12,13],[4,3,2]], [[2,10,3]], [[8,12,18]]], #3 obj
                 [[[2,10,17,6]], [[10,18,16,13]], [[3,17,8,12]]], #4 obj                  
                #   [[('right',0,4)], [('close',4,3)], [('above',1,2),('dist',0,3,7)], [('close',1,4),('close',2,3)], [('right',0,2),('dist',4,2,9),('close',3,2)]], #5 obj
]                  
test_shapes = [[[[0, 0, 1]], [[0, 1, 2]], [[4, 1, 3]], [[2, 3, 3]], [[2, 0, 4],[2, 0, 4]], [[0, 0, 1]], [[3, 1, 0]]],  # 3 obj
               [[[2,3,2,0]], [[3,2,4,1]], [[1,0,0,0]]], #4 obj
    #   [[('right',0,4)], [('close',4,3)], [('above',1,2),('dist',0,3,7)], [('close',1,4),('close',2,3)], [('right',0,2),('dist',4,2,9),('close',3,2)]], #5 obj
]
test_colors = [[[[1, 2, 3]], [[4, 0, 1]], [[0, 0, 2]], [[2, 3, 1]], [[4, 1, 3],[4, 1, 3]], [[1, 2, 3]], [[0, 1, 2]]],  # 3 obj
              [[[3,2,1,4]], [[0,1,2,0]], [[4,3,0,1]]], #4 obj
    #   [[('right',0,4)], [('close',4,3)], [('above',1,2),('dist',0,3,7)], [('close',1,4),('close',2,3)], [('right',0,2),('dist',4,2,9),('close',3,2)]], #5 obj
]                        

for i_obj,n_obj in enumerate([3,4]): #,4,5]: 
    base_G = np.array([[[1,0] for _ in range(n_obj)] for _ in range(n_obj)]) #diag detached (change for orientation possibly) 
      
    for train_scenario_idx in range(len(all_predicates[i_obj])): #5 train scenarios for each number of objects
        task_name = os.path.join(args.task_dir,'train_nobj{}_taskid{}'.format(str(n_obj),str(train_scenario_idx)))
        
        predicates = all_predicates[i_obj][train_scenario_idx] 
        task_G = base_G.copy()
        for pred in predicates:
            i,j = pred[1], pred[2]
            task_G[i,j] = [0,1]
            task_G[j,i] = [0,1]

        expert_subgoals = all_expert_subgoals[i_obj][train_scenario_idx]

        task_info = {
            'G': task_G,
            'predicates': predicates,
            'expert_subgoals': expert_subgoals,
            'init_positions': train_init_pos[i_obj][train_scenario_idx],
            'shapes': train_shapes[i_obj][train_scenario_idx],
            'colors': train_colors[i_obj][train_scenario_idx],
            'num_agents': n_obj,
        }
        
        print(task_name)
        print(task_info)

        with open(task_name + ".pkl", "wb") as f:
            pickle.dump(task_info, f)

        for test_scenario_idx in range(
            len(test_colors[i_obj][train_scenario_idx])
        ):  # 5 test scenarios for each train scenario
            task_name = os.path.join(
                args.task_dir,
                "test_nobj{}_taskid{}_{}".format(
                    str(n_obj), str(train_scenario_idx), str(test_scenario_idx)
                ),
            )

            n_test_obj = len(test_colors[i_obj][train_scenario_idx][test_scenario_idx])

            if n_obj != n_test_obj: #different G (different # of objs)
                test_base_G = np.array([[[1,0] for _ in range(n_test_obj)] for _ in range(n_test_obj)])          
                task_G = test_base_G.copy()
                for pred in predicates:
                    i,j = pred[1], pred[2]
                    task_G[i,j] = [0,1]
                    task_G[j,i] = [0,1]

            task_info = {
                'G': task_G,
                'predicates': predicates,
                'expert_subgoals': expert_subgoals,
                'init_positions': test_init_pos[i_obj][train_scenario_idx][test_scenario_idx],
                'shapes': test_shapes[i_obj][train_scenario_idx][test_scenario_idx],
                'colors': test_colors[i_obj][train_scenario_idx][test_scenario_idx],
                'num_agents': n_test_obj,
            }  

            print(task_name)
            print(task_info)                                          

            with open(task_name + ".pkl", "wb") as f:
                pickle.dump(task_info, f)
