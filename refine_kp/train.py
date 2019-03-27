import tensorflow as tf
from functools import reduce
import numpy as np

from refine_kp.aug_data_loader import data_loader_pre as data_loader, list_to_kp_map, kp_map_to_list, display, CenterGaussianHeatMap
from matplotlib import pyplot as plt
from PIL import Image
from refine_kp.cpm_model import cpm_construct
from uuid import uuid1
from collections import namedtuple

def retrieve_conclusion(heatmap_producer, mat = 8):
    #[batch, 64, 64, 17]
    def max_xy(input):
        h_dim, w_dim = tf.shape(input)[0], tf.shape(input)[1]
        max_idx = tf.argmax(tf.reshape(input, [-1]), axis=-1,
                            output_type=tf.int32)

        x = tf.cast(tf.divide(max_idx, h_dim), tf.int32)
        y = tf.cast(max_idx - x * h_dim, tf.int32)
        x = x * mat
        y = y * mat

        return tf.stack([y, x], axis=0)

    #### [-1, h, w]
    flatten_input = tf.reshape(tf.transpose(heatmap_producer, [0, 3, 1, 2]), [-1] + list(map(int, heatmap_producer.get_shape()[1:3])))
    flatten_idx_conclusion = tf.map_fn(max_xy, flatten_input, dtype=tf.int32)

    return tf.reshape(flatten_idx_conclusion, [-1, 17, 2])


#### gt heatmap are dense gaussian, gt C are sparse like
def total_encoder_decoder_builder():
    cfg = namedtuple("cfg" ,["output_shape", "input_shape",
                             "batch_size", "num_kps", "use_loss"])
    cfg.input_shape = [256, 256]
    cfg.output_shape = [int(256 / 8), int(256 / 8)]
    cfg.batch_size = 16
    cfg.num_kps = 17
    cfg.use_loss = "refine"

    input, gt_heatmap, cpm_m, output, gt_C, mask = cpm_construct(cfg)
    total_loss = cpm_m.total_loss

    #### [batch, 17, 2]
    pred_hm = retrieve_conclusion(heatmap_producer=output)
    gt_hm = retrieve_conclusion(heatmap_producer=input[...,3:], mat = 1)

    opt = tf.train.AdamOptimizer(0.0001).minimize(total_loss)
    return (input, gt_C, gt_heatmap, mask ,total_loss, opt, pred_hm, gt_hm, cfg)

def single_valid(cfg ,image, kp_map, resize_to = None, debug = False, type = "true"):
    assert resize_to is not None
    kp_map_list = kp_map_to_list(kp_map)

    w, h = resize_to
    image = Image.fromarray(image.astype(np.uint8))
    ori_w, ori_h = image.size
    image = image.resize((w, h))

    #### person list of [17, 3] produce maask for each by last
    kp_map_list = list(map(lambda kp_: np.concatenate([kp_[:, 0:1].astype(np.float32) / ori_w * w / 1, kp_[:, 1:2].astype(np.float32) / ori_h * h / 1,
                                                       kp_[:, 2:3]], axis=-1).astype(np.int32), kp_map_list))

    #### person list of [h, w, 17]
    kp_heatmap = list(map(lambda single_person_map: reduce(lambda a, b: np.concatenate([a, b], axis=-1),map(lambda idx:(np.zeros(shape=(int(h/ 1), int(w / 1))) if single_person_map[idx][-1] == 0
                                                                                                                        else CenterGaussianHeatMap(img_height = int(h / 1), img_width = int(w / 1),
                                                                                                                                                   c_x = single_person_map[idx][0], c_y = single_person_map[idx][1])).reshape([int(h / 1), int(w / 1), 1]),
                                                                                                            range(len(single_person_map)))), kp_map_list))

    image = np.asarray(image).astype(np.float32)
    if debug:
        kp_map = list_to_kp_map(input_list = kp_map_list, h = int(h / 1), w = int(w / 1))
        plt.imshow(display.summary_skeleton(image, kp_map))
        pic_path = r"C:\Coding\Python\multi_person_pose_refine\{}_pics\{}_{}.png".format(cfg.use_loss ,type ,uuid1())
        plt.savefig(pic_path)
        print("have saved : {}".format(pic_path))

    return (image, kp_map_list, kp_heatmap)


def train():
    input, gt_C, gt_heatmap, mask ,total_loss, opt, pred_hm, gt_hm, cfg = total_encoder_decoder_builder()
    saver = tf.train.Saver()

    train_iter = data_loader(dataType="train")
    test_iter = data_loader(dataType="test")

    with tf.Session() as sess:
        #sess.run(tf.global_variables_initializer())
        saver.restore(sess=sess, save_path="C:\\Coding\\Python\\RefineNet\\model_dir\\model_{}_15500".format(cfg.use_loss))

        for i in range(int(1e6)):
            batch_input, batch_kp_heatmap_true, batch_kp_map_true, batch_kp_map_true_mask = train_iter.__next__()

            _, train_total_loss, train_hm = sess.run(
                [opt, total_loss, pred_hm], feed_dict={
                    input: batch_input,
                    gt_C: batch_kp_map_true,
                    gt_heatmap: batch_kp_heatmap_true,
                    mask: batch_kp_map_true_mask
                }
            )

            if i % 50 == 0:
                print("train_loss : {}".format(train_total_loss))

            if i % 50 == 0 and i > 0:
                batch_input, batch_kp_heatmap_true, batch_kp_map_true, batch_kp_map_true_mask = test_iter.__next__()
                valid_gt_hm ,valid_total_loss, valid_hm = sess.run(
                    [gt_hm ,total_loss, pred_hm], feed_dict={
                        input: batch_input,
                        gt_C: batch_kp_map_true,
                        gt_heatmap: batch_kp_heatmap_true,
                        mask: batch_kp_map_true_mask
                    }
                )
                print("valid_loss : {}".format(valid_total_loss))

                kp_map = list_to_kp_map(input_list = [np.concatenate([valid_gt_hm[0], np.ones([17, 1])], axis=-1).astype(np.int32)], h = 256, w = 256)
                single_valid(cfg ,batch_input[0][..., :3], kp_map = kp_map,
                             resize_to = (256, 256), debug = True, type = "true")
                print("continue !")

                kp_map = list_to_kp_map(input_list = [np.concatenate([valid_hm[0], np.ones([17, 1])], axis=-1).astype(np.int32)], h = 256, w = 256)
                single_valid(cfg ,batch_input[0][..., :3], kp_map = kp_map,
                             resize_to = (256, 256), debug = True, type = "pred")
                print("continue !")

                if i % 500 == 0:
                    saver.save(sess=sess, save_path=r"C:\Coding\Python\RefineNet\model_dir\model_{}_{}".format(cfg.use_loss ,i + 15500))
                    pass


    pass

if __name__ == "__main__":
    train()

    pass