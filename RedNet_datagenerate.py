import argparse
import os
import time
import torch

import numpy as np


from PIL import Image
import matplotlib.pyplot as plt
from constants import mp3d_category_id

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

cv2 = try_cv2_import()

img_dir_train_file = '/data/p305574/RedNet/data/img_dir_train.txt'
depth_dir_train_file = '/data/p305574/RedNet/data/depth_dir_train.txt'
label_dir_train_file = '/data/p305574/RedNet/data/label_train.txt'
img_dir_test_file = '/data/p305574/RedNet/data/img_dir_test.txt'
depth_dir_test_file = '/data/p305574/RedNet/data/depth_dir_test.txt'
label_dir_test_file = '/data/p305574/RedNet/data/label_test.txt'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
image_w = 640
image_h = 480

######## matterport_category_mappings ######
fileName = 'configs/matterport_category_mappings.tsv'

text = ''
lines = []
items = []
hm3d_semantic_mapping={}

with open(fileName, 'r') as f:
    text = f.read()
lines = text.split('\n')

for l in lines:
    items.append(l.split('    '))

for i in items:
    if len(i) > 3:
        hm3d_semantic_mapping[i[2]] = i[-1]
####################



def print_scene_recur(scene, limit_output=10):
    count = 0
    # for level in scene.levels:
    #     print(
    #         f"Level id:{level.id}, center:{level.aabb.center},"
    #         f" dims:{level.aabb.sizes}"
    #     )
    #     for region in level.regions:
    #         print(
    #             f"Region id:{region.id}, category:{region.category.name()},"
    #             f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            # )
    for obj in scene.objects:
        print(
            f"Object id:{obj.id}, category:{obj.category.name()},"
            f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
        )
        print(obj.id.split("_")[-1])
        count += 1
        if count >= limit_output:
            return None

def train():

    config=habitat.get_config("configs/rednet_hm3d.yaml")

    env = habitat.Env(
        config=config
    )
    # dataset = habitat.make_dataset(config.DATASET.TYPE)
    # scenes = config.DATASET.CONTENT_SCENES
    # print("************************************* scenes lens:", len(scenes))
    env.seed(10000)
    # observations = env.reset()
    # print(observations)

    old_scene_id = ''
    old_category = ''

    num_train = len(env.episodes)
    print("num_train: ", num_train)

    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.SIMULATOR.FORWARD_STEP_SIZE
    follower = ShortestPathFollower(
        env.sim, goal_radius, False
    )
    count_steps = 0
    for i in range(num_train):

        observations = env.reset()

        scene_id = env.current_episode.scene_id
        category = env.current_episode.goals[0].object_category

        # while scene_id == old_scene_id:
        #     observations = env.reset()

        #     scene_id = env.current_episode.scene_id
        #     category = env.current_episode.goals[0].object_category

        old_scene_id = scene_id
        old_category = category

        scene = env.sim.semantic_annotations()
        # print_scene_recur(scene, limit_output=100)

        # obj_map = {}
        # for obj in scene.objects:
        #    obj_map[int(obj.id.split("_")[-1])] = obj.id

        # print(mp3d_category_id)

        # for k,v in mp3d_category_id[0]:
        #     print("k: ", k)
        #     print("v: ", v)
        skip_step = 0
        while not env.episode_over:
            best_action = follower.get_next_action(
                env.current_episode.goals[0].position
            )
            if best_action is None:
                break

            observations = env.step(best_action)
            skip_step += 1
   
            semantic = observations['semantic']
            # print("semantic: ", semantic.shape)

            se = list(set(semantic.ravel()))
            # print(se)

            count_sem = 0
            for i in range(len(se)):
                # print("category: ", scene.objects[se[i]].category.name())

                if scene.objects[se[i]].category.name() in hm3d_semantic_mapping:
                    hm3d_category_name = hm3d_semantic_mapping[scene.objects[se[i]].category.name()]
                else:
                    hm3d_category_name = scene.objects[se[i]].category.name()

                if hm3d_category_name in mp3d_category_id:
                    # print("sum: ", np.sum(sem_output[sem_output==se[i]])/se[i])
                    count_sem += np.sum(semantic[semantic==se[i]])/se[i]
                    semantic[semantic==se[i]] = mp3d_category_id[hm3d_category_name]
                else :
                    semantic[
                        semantic==se[i]
                        ] = 1
            # print("count_sem: ", count_sem)

            
            if count_sem > 2000 and skip_step > 3:
                fn = '/data/p305574/RedNet/data/rgb/Vis-rgb-{}.png'.format(count_steps)
                cv2.imwrite(fn, observations['rgb'])
                with open(img_dir_train_file, 'a') as f:
                    f.write(fn+'\n')

                fn = '/data/p305574/RedNet/data/depth/Vis-depth-{}.png'.format(count_steps)
                cv2.imwrite(fn, observations['depth'])
                with open(depth_dir_train_file, 'a') as f:
                    f.write(fn+'\n')

                fn = '/data/p305574/RedNet/data/semantic/Vis-sem-{}.npy'.format(count_steps)
                np.save(fn, semantic)
                with open(label_dir_train_file, 'a') as f:
                    f.write(fn+'\n')

                count_steps +=1
                skip_step = 0

                if count_steps % 500 == 0:
                    print("count: ", count_steps)
                    
        if count_steps > 100000:
            break

            

    print("Data completed ")


if __name__ == '__main__':

    train()
