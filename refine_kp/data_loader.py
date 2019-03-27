import numpy as np
from glob import glob
import pickle
from random import choice
import sklearn
from PIL import Image

import math
from personlab import display
from uuid import uuid1
from matplotlib import pyplot as plt
from functools import reduce
from collections import defaultdict

def single_valid_pre(image, kp_map, resize_to = None, debug = False):
    assert resize_to is not None
    kp_map_list = kp_map_to_list(kp_map)

    w, h = resize_to
    image = Image.fromarray(image.astype(np.uint8))
    ori_w, ori_h = image.size
    image = image.resize((w, h))

    #### person list of [17, 3] produce maask for each by last
    kp_map_list = list(map(lambda kp_: np.concatenate([kp_[:, 0:1].astype(np.float32) / ori_w * w, kp_[:, 1:2].astype(np.float32) / ori_h * h,
                                                       kp_[:, 2:3]], axis=-1).astype(np.int32), kp_map_list))

    #### person list of [h, w, 17]
    kp_heatmap = list(map(lambda single_person_map: reduce(lambda a, b: np.concatenate([a, b], axis=-1),map(lambda idx:(np.zeros(shape=(int(h), int(w))) if single_person_map[idx][-1] == 0
                                                                                                                        else CenterGaussianHeatMap(img_height = int(h), img_width = int(w),
                                                                                                                                                   c_x = single_person_map[idx][0], c_y = single_person_map[idx][1])).reshape([int(h), int(w), 1]),
                                                                                                            range(len(single_person_map)))), kp_map_list))

    image = np.asarray(image).astype(np.float32)

    if debug:
        kp_map = list_to_kp_map(input_list = kp_map_list, h = int(h), w = int(w))
        plt.imshow(display.summary_skeleton(image, kp_map))
        #plt.show()
        pic_path = r"C:\Coding\Python\RefineNet\pics\{}.png".format(uuid1())
        plt.savefig(pic_path)
        print("have saved : {}".format(pic_path))

    return (image, kp_map_list, kp_heatmap)

def CenterGaussianHeatMap(img_height, img_width, c_x, c_y, std = 1.0):
    times = 2
    gaussian_map = np.zeros((img_height, img_width))
    for x_p in range(max(int(c_x - times * std), 0), min(int(c_x + times * std), img_width)):
        for y_p in range(max(int(c_y - times * std), 0), min(int(c_y + times * std), img_height)):
            gaussian_map[y_p, x_p] = math.exp(-1 * (((x_p - c_x) * (x_p - c_x) +
                                                     (y_p - c_y) * (y_p - c_y)) / 2.0 / std / std))
    return gaussian_map

KP_NAMES = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle'
]
NUM_KP = len(KP_NAMES)

#### [h, w, 2] to [17, 3]
def kp_map_to_list(kp_map):
    num_people = kp_map[..., 0].max()
    kp_people_nest_list = []

    for p_i in range(num_people):
        kp_point = np.zeros([NUM_KP, 3], dtype=np.int16)
        for y, x in zip(*np.nonzero(kp_map[..., 0] == p_i + 1)):
            kp_i = kp_map[y, x, 1]
            kp_point[kp_i-1] = (x, y, 1)
        kp_people_nest_list.append(kp_point)

    return kp_people_nest_list

#### [17, 3] to [h, w, 2]
def list_to_kp_map(input_list, h, w):
    need_array = np.zeros((h, w, 2), dtype=np.int32)
    for p_i, kp_point in enumerate(input_list):
        for kp_i ,(x, y, index) in enumerate(kp_point):
            if index == 1:
                need_array[y, x, 1] = kp_i + 1
                need_array[y, x, 0] = p_i + 1
    return need_array

#### single input shape [h, w, c] [17, 3]
def single_valid(image, kp_map, resize_to = None, debug = False):
    assert resize_to is not None
    kp_map_list = kp_map_to_list(kp_map)

    w, h = resize_to
    image = Image.fromarray(image.astype(np.uint8))
    ori_w, ori_h = image.size
    image = image.resize((w, h))

    #### person list of [17, 3] produce maask for each by last
    kp_map_list = list(map(lambda kp_: np.concatenate([kp_[:, 0:1].astype(np.float32) / ori_w * w / 8, kp_[:, 1:2].astype(np.float32) / ori_h * h / 8,
                                                       kp_[:, 2:3]], axis=-1).astype(np.int32), kp_map_list))

    #### person list of [h, w, 17]
    kp_heatmap = list(map(lambda single_person_map: reduce(lambda a, b: np.concatenate([a, b], axis=-1),map(lambda idx:(np.zeros(shape=(int(h/ 8), int(w / 8))) if single_person_map[idx][-1] == 0
                                                                                                                        else CenterGaussianHeatMap(img_height = int(h / 8), img_width = int(w / 8),
                                                                                                                                                   c_x = single_person_map[idx][0], c_y = single_person_map[idx][1])).reshape([int(h / 8), int(w / 8), 1]),
                                                                                                            range(len(single_person_map)))), kp_map_list))

    image = np.asarray(image).astype(np.float32)

    if debug:
        kp_map = list_to_kp_map(input_list = kp_map_list, h = int(h / 8), w = int(w / 8))
        plt.imshow(display.summary_skeleton(image, kp_map))
        #plt.show()
        pic_path = r"C:\Coding\Python\RefineNet\pics\{}.png".format(uuid1())
        plt.savefig(pic_path)
        print("have saved : {}".format(pic_path))

    return (image, kp_map_list, kp_heatmap)


def single_retrieve(input_file):
    with open(input_file, "rb") as f:
        pickled_dict = pickle.load(f)
    image = pickled_dict["image"]
    kp_map_true = pickled_dict["kp_map_true"]
    kp_map_pred = pickled_dict["kp_map_pred"]
    return (image, kp_map_true, kp_map_pred)

def true_pred_indices_alignment(true_kp_map_list, pre_kp_map_list, filter_num = 4):
    #### list of [17]
    filter_keys = list(filter(lambda i: true_kp_map_list[i][:, -1].sum() >= filter_num, range(len(true_kp_map_list))))
    filter_values = list(filter(lambda i: pre_kp_map_list[i][:, -1].sum() >= filter_num, range(len(pre_kp_map_list))))

    def sort_inner(input):
        input_list = input.tolist()
        input_list.sort()
        return np.asarray(input_list)

    true_kp_map_list = list(map(sort_inner, true_kp_map_list))
    pre_kp_map_list = list(map(sort_inner, pre_kp_map_list))

    look_up_array = np.zeros(shape=[len(true_kp_map_list), len(pre_kp_map_list)])
    for r_i in range(len(true_kp_map_list)):
        r_person = true_kp_map_list[r_i]
        for c_i in range(len(pre_kp_map_list)):
            c_person = pre_kp_map_list[c_i]
            look_up_array[r_i][c_i] = ((r_person - c_person) ** 2).reshape([-1]).sum()

    def re_sort_inner(input):
        input_list = input.tolist()
        req = []
        for x, y, c in input_list:
            req.append([y, x, c])
        req.sort()
        return np.asarray(req)

    true_kp_map_list = list(map(re_sort_inner, true_kp_map_list))
    pre_kp_map_list = list(map(re_sort_inner, pre_kp_map_list))

    for r_i in range(len(true_kp_map_list)):
        r_person = true_kp_map_list[r_i]
        for c_i in range(len(pre_kp_map_list)):
            c_person = pre_kp_map_list[c_i]
            look_up_array[r_i][c_i] += ((r_person - c_person) ** 2).reshape([-1]).sum()

    m_val = look_up_array.mean()

    row_conclusion = defaultdict(list)

    for x, y in  zip(*(np.where(look_up_array < 4000))):
        row_conclusion[x].append((look_up_array[x][y], y))

    indices_dict = dict()
    for x in row_conclusion.keys():
        indices_dict[x] = list(map(lambda tt2: tt2[-1],sorted(row_conclusion[x],key= lambda t2: t2[0])))[0]

    req = dict()
    for k, v in indices_dict.items():
        if k in filter_keys and v in filter_values:
            req[k] = v

    return (m_val ,req)

#### true_pred_indices_alignment(train_kp_map_list, pre_kp_map_list)
def apply_alignment_indices(true_kp_map_list, pre_kp_map_list, indices_dict):
    req_true, req_pre = [], []
    for true_indice, pre_indice in indices_dict.items():
        req_true.append(true_kp_map_list[true_indice])
        req_pre.append(pre_kp_map_list[pre_indice])
    return (req_true, req_pre)

def data_loader_pre(batch_size = 16, dataType = "train", resize_to = (256, 256)):
    assert dataType in ["train", "test"]
    #data_path = r"E:\Temp\conclusion_dir_tiny" if dataType == "train" else r"E:\Temp\valid_dir"
    data_path = r"E:\Temp\conclusion_dir" if dataType == "train" else r"E:\Temp\valid_dir"
    w, h = resize_to

    times = 0
    batch_input = np.zeros(shape=[batch_size, h, w, 3 + 17])
    batch_kp_heatmap_true = np.zeros(shape=[batch_size, int(h / 8), int(w / 8), 17])
    batch_kp_map_true, batch_kp_map_true_mask = np.zeros(shape=[batch_size, 17, 2]), np.zeros(shape=[batch_size, 17])

    while True:
        print("start new pkl :")

        single_pkl_path = choice(glob(r"{}\*".format(data_path)))
        images, kp_map_true, kp_map_pre = single_retrieve(single_pkl_path)
        images, kp_map_true, kp_map_pre = sklearn.utils.shuffle(images, kp_map_true, kp_map_pre)

        for start in range(0 ,len(images), batch_size):
            end = min(start + batch_size, len(images))
            for index in range(start, end):
                if images[index].shape[-1] != 3:
                    print("skip shape {}".format(images[index].shape))
                    continue

                image, kp_map_list_true, kp_heatmap_true = single_valid(images[index], kp_map_true[index], resize_to=resize_to)
                _ ,kp_map_list_pre, kp_heatmap_pre = single_valid_pre(images[index], kp_map_pre[index], resize_to=resize_to)

                kp_map_list_pre_ = list(map(lambda x: np.concatenate([(x[..., :2] / 8).astype(np.int32), x[...,2:]], axis=-1), kp_map_list_pre))
                m_val ,indices_dict = true_pred_indices_alignment(kp_map_list_true, kp_map_list_pre_)
                if not(m_val < int(1e10)):
                    continue

                kp_map_list_true, kp_map_list_pre =  apply_alignment_indices(true_kp_map_list = kp_map_list_true, pre_kp_map_list = kp_map_list_pre, indices_dict=indices_dict)
                kp_heatmap_true, kp_heatmap_pre =  apply_alignment_indices(true_kp_map_list = kp_heatmap_true, pre_kp_map_list = kp_heatmap_pre, indices_dict=indices_dict)

                assert len(kp_map_list_true) == len(kp_heatmap_true) and len(kp_map_list_pre) == len(kp_heatmap_pre) and len(kp_map_list_true) == len(kp_heatmap_pre)

                for person_index in range(len(kp_map_list_true)):
                    kp_map_s_true = kp_map_list_true[person_index]

                    #### [h, w, 17]
                    kp_heatmap_s_true = kp_heatmap_true[person_index]
                    kp_heatmap_s_pre = kp_heatmap_pre[person_index]

                    #### [h, w, 3 + 17]
                    input = np.concatenate([image, kp_heatmap_s_pre], axis=-1)
                    #### [17, 2] [17]
                    kp_map_s_true, kp_map_s_true_mask = kp_map_s_true[...,:2], kp_map_s_true[...,2]

                    batch_input[times % batch_size] = input
                    batch_kp_heatmap_true[times % batch_size] = kp_heatmap_s_true
                    batch_kp_map_true[times % batch_size] = kp_map_s_true
                    batch_kp_map_true_mask[times % batch_size] = kp_map_s_true_mask

                    times += 1
                    if times % batch_size == 0:
                        batch_input, batch_kp_heatmap_true, batch_kp_map_true, batch_kp_map_true_mask = sklearn.utils.shuffle(batch_input, batch_kp_heatmap_true, batch_kp_map_true, batch_kp_map_true_mask)
                        yield (batch_input, batch_kp_heatmap_true, batch_kp_map_true, batch_kp_map_true_mask)

                        batch_input = np.zeros(shape=[batch_size, h, w, 3 + 17])
                        batch_kp_heatmap_true = np.zeros(shape=[batch_size, int(h / 8), int(w / 8), 17])
                        batch_kp_map_true, batch_kp_map_true_mask = np.zeros(shape=[batch_size, 17, 2]), np.zeros(shape=[batch_size, 17])

                    if times % int(1e2) == 0:
                        print("yield samples {}".format(times))

def data_loader_test():
    iter = data_loader_pre()

    req_list = []
    while True:
        item_t = iter.__next__()


if __name__ == "__main__":
    data_loader_test()
