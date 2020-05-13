import pandas as pd
import numpy as np
import os
import re
import json
from tqdm import tqdm
import cv2
os.chdir('../')


DIR_INPUT = '/kaggle/input/global-wheat-detection'
DIR_TRAIN = f'{DIR_INPUT}/train'
DIR_TEST = f'{DIR_INPUT}/test'

class WheatDataset():
    def __init__(self, rootpath_data):
        self.rootpath_data = rootpath_data

    def format_coco(self, df: pd.DataFrame):
        "i_id: image's id; b_id: box's id"
        "note the image_id !"
        assert isinstance(df, pd.DataFrame), "input data is not a instance of pd.df"

        # init lists
        images = []
        annotations = []
        categories = []

        # df to array
        boxes = df[['x', 'y', 'w', 'h']].values  # N * 4
        images_names = df[['image_id']]  # N * 1  image_id is image name
        # print(images_names.shape)
        dropped_image_names = images_names.drop_duplicates()  # num_images * 1
        images_sizes = df[['height', 'width']].values  # N * 2
        boxes_ids = 0  # all the positive label: only one kind of label;
                       # ???In coco: 0 is also as a real class, not background???.

        num_samples = len(boxes)
        assert len(dropped_image_names.index) < num_samples, "image id has repeated!!!"
        i_id = 0
        for n in tqdm(range(num_samples)):
            b_id = n
            if i_id < len(dropped_image_names.index) and images_names.iloc[n].values == dropped_image_names.iloc[i_id].values:
                image = {'file_name': f'{images_names.iloc[n].values[0]}.jpg',
                         'height': int(images_sizes[n, 0]),
                         'width': int(images_sizes[n, 1]),
                         'id': i_id + 1
                         }
                images.append(image)
                i_id += 1

            annotation = {'segmentation': [],  # if you have mask labels
                          'area': int(boxes[n, 2]) * int(boxes[n, 3]),
                          'iscrowd': 0,
                          'image_id': i_id,  # match image's id
                          'bbox': boxes[n, :].tolist(),
                          'category_id': boxes_ids,
                          'id': b_id + 1}
            annotations.append(annotation)

        categories.append({'id': 0, 'name': 'wheat'})

        print("dump size:", len(images), len(annotations), len(categories))
        return {'images': images, 'annotations': annotations, 'categories': categories}


    def convert_dataset(self, train_df, image_ids):
        valid_ids = image_ids[-665:]
        train_ids = image_ids[:-665]
        valid_df = train_df[train_df['image_id'].isin(valid_ids)]
        train_df = train_df[train_df['image_id'].isin(train_ids)]
        # print(f'{valid_df.shape}, {train_df.shape}')
        with open(f"{self.rootpath_data}/train.json", "w") as json_file:
            json.dump(self.format_coco(train_df), json_file)
        with open(f"{self.rootpath_data}/valid.json", "w") as json_file:
            json.dump(self.format_coco(valid_df), json_file)

        print("convert finished !!!")

    def load_dataset(self):
        train_df = pd.read_csv(f'{self.rootpath_data}/train.csv')
        # print("train shape: ", train_df.shape)
        train_df['x'] = -1
        train_df['y'] = -1
        train_df['w'] = -1
        train_df['h'] = -1
        def expand_bbox(x):
            r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
            if len(r) == 0:
                r = [-1, -1, -1, -1]
            return r

        train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
        train_df.drop(columns=['bbox'], inplace=True)
        train_df['x'] = train_df['x'].astype(np.float)
        train_df['y'] = train_df['y'].astype(np.float)
        train_df['w'] = train_df['w'].astype(np.float)
        train_df['h'] = train_df['h'].astype(np.float)
        image_ids = train_df['image_id'].unique()  # unique return ndarray
        return train_df, image_ids

    def get_normalized_cfgs(self):
        _, image_ids = self.load_dataset()
        image_ids = image_ids.tolist()
        means = []
        stds = []
        process_bar = tqdm(range(len(image_ids)))
        for i in process_bar:
            process_bar.set_description('')
            image = cv2.imread(f'{self.rootpath_data}/train/{image_ids[i]}.jpg', cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
            mean = [np.mean(image[..., 0]), np.mean(image[..., 1]), np.mean(image[..., 2])]
            std = [np.std(image[..., 0]), np.std(image[..., 1]), np.std(image[..., 2])]
            means.append(mean)
            stds.append(std)
        process_bar.close()
        global_mean = np.asarray(means).mean(axis=0).tolist()
        global_std = np.asarray(stds).mean(axis=0).tolist()
        return global_mean, global_std

if __name__ == '__main__':
    wheat_dataset = WheatDataset('/root/atticus/wheat')
    # mean, std = wheat_dataset.get_normalized_cfgs()
    # print(mean, std)
    # >>> [80.2324447631836, 80.93988037109375, 54.676353454589844]
    # >>> [53.057960510253906, 53.754241943359375, 45.067726135253906]

    data = wheat_dataset.load_dataset()
    wheat_dataset.convert_dataset(*data)
