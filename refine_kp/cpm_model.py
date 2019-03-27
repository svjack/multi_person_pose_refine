import tensorflow as tf
import pickle

import pause

class CPM_Model(object):
    def __init__(self, stages, joints):
        self.stages = stages
        self.stage_heatmap = []
        self.stage_loss = [0] * stages
        self.total_loss = 0
        self.input_image = None
        self.center_map = None
        self.gt_heatmap = None
        self.learning_rate = 0
        self.merged_summary = None
        self.joints = joints
        self.batch_size = 0

    def build_model(self, input_image, center_map, batch_size):
        self.batch_size = batch_size
        self.input_image = input_image
        self.center_map = center_map
        with tf.variable_scope('pooled_center_map'):
            self.center_map = tf.layers.average_pooling2d(inputs=self.center_map,
                                                          pool_size=[9, 9],
                                                          strides=[8, 8],
                                                          padding='same',
                                                          name='center_map')
        with tf.variable_scope('sub_stages'):
            sub_conv1 = tf.layers.conv2d(inputs=input_image,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv1')
            sub_conv2 = tf.layers.conv2d(inputs=sub_conv1,
                                         filters=64,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv2')
            sub_pool1 = tf.layers.max_pooling2d(inputs=sub_conv2,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='sub_pool1')
            sub_conv3 = tf.layers.conv2d(inputs=sub_pool1,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv3')
            sub_conv4 = tf.layers.conv2d(inputs=sub_conv3,
                                         filters=128,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv4')
            sub_pool2 = tf.layers.max_pooling2d(inputs=sub_conv4,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='sub_pool2')
            sub_conv5 = tf.layers.conv2d(inputs=sub_pool2,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv5')
            sub_conv6 = tf.layers.conv2d(inputs=sub_conv5,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv6')
            sub_conv7 = tf.layers.conv2d(inputs=sub_conv6,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv7')
            sub_conv8 = tf.layers.conv2d(inputs=sub_conv7,
                                         filters=256,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv8')
            sub_pool3 = tf.layers.max_pooling2d(inputs=sub_conv8,
                                                pool_size=[2, 2],
                                                strides=2,
                                                padding='same',
                                                name='sub_pool3')
            sub_conv9 = tf.layers.conv2d(inputs=sub_pool3,
                                         filters=512,
                                         kernel_size=[3, 3],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='sub_conv9')
            sub_conv10 = tf.layers.conv2d(inputs=sub_conv9,
                                          filters=512,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv10')
            sub_conv11 = tf.layers.conv2d(inputs=sub_conv10,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv11')
            sub_conv12 = tf.layers.conv2d(inputs=sub_conv11,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv12')
            sub_conv13 = tf.layers.conv2d(inputs=sub_conv12,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv13')
            sub_conv14 = tf.layers.conv2d(inputs=sub_conv13,
                                          filters=256,
                                          kernel_size=[3, 3],
                                          strides=[1, 1],
                                          padding='same',
                                          activation=tf.nn.relu,
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          name='sub_conv14')

            self.sub_stage_img_feature = tf.layers.conv2d(inputs=sub_conv14,
                                                          filters=128,
                                                          kernel_size=[3, 3],
                                                          strides=[1, 1],
                                                          padding='same',
                                                          activation=tf.nn.relu,
                                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                          name='sub_stage_img_feature')

        with tf.variable_scope('stage_1'):
            conv1 = tf.layers.conv2d(inputs=self.sub_stage_img_feature,
                                     filters=512,
                                     kernel_size=[1, 1],
                                     strides=[1, 1],
                                     padding='same',
                                     activation=tf.nn.relu,
                                     kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                     name='conv1')
            self.stage_heatmap.append(tf.layers.conv2d(inputs=conv1,
                                                       filters=self.joints,
                                                       kernel_size=[1, 1],
                                                       strides=[1, 1],
                                                       padding='same',
                                                       kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                       name='stage_heatmap'))
        for stage in range(2, self.stages + 1):
            self._middle_conv(stage)

    def _middle_conv(self, stage):
        with tf.variable_scope('stage_' + str(stage)):
            self.current_featuremap = tf.concat([self.stage_heatmap[stage - 2],
                                                 self.sub_stage_img_feature,
                                                 self.center_map,
                                                 ],
                                                axis=3)
            mid_conv1 = tf.layers.conv2d(inputs=self.current_featuremap,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv1')
            mid_conv2 = tf.layers.conv2d(inputs=mid_conv1,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv2')
            mid_conv3 = tf.layers.conv2d(inputs=mid_conv2,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv3')
            mid_conv4 = tf.layers.conv2d(inputs=mid_conv3,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv4')
            mid_conv5 = tf.layers.conv2d(inputs=mid_conv4,
                                         filters=128,
                                         kernel_size=[7, 7],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv5')
            mid_conv6 = tf.layers.conv2d(inputs=mid_conv5,
                                         filters=128,
                                         kernel_size=[1, 1],
                                         strides=[1, 1],
                                         padding='same',
                                         activation=tf.nn.relu,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name='mid_conv6')
            self.current_heatmap = tf.layers.conv2d(inputs=mid_conv6,
                                                    filters=self.joints,
                                                    kernel_size=[1, 1],
                                                    strides=[1, 1],
                                                    padding='same',
                                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                    name='mid_conv7')
            self.stage_heatmap.append(self.current_heatmap)

    def extract_coordinate(self, heatmap_outs, cfg):
        shape = heatmap_outs.get_shape().as_list()
        batch_size = tf.shape(heatmap_outs)[0]
        height = shape[1]
        width = shape[2]
        output_shape = (height, width)

        # coordinate extract from output heatmap
        y = [i for i in range(output_shape[0])]
        x = [i for i in range(output_shape[1])]
        xx, yy = tf.meshgrid(x, y)
        xx = tf.to_float(xx) + 1
        yy = tf.to_float(yy) + 1

        heatmap_outs = tf.reshape(tf.transpose(heatmap_outs, [0, 3, 1, 2]), [batch_size, cfg.num_kps, -1])
        heatmap_outs = tf.nn.softmax(heatmap_outs)
        heatmap_outs = tf.transpose(tf.reshape(heatmap_outs, [batch_size, cfg.num_kps, output_shape[0], output_shape[1]]), [0, 2, 3, 1])

        x_out = tf.reduce_sum(tf.multiply(heatmap_outs, tf.tile(tf.reshape(xx,[1, output_shape[0], output_shape[1], 1]), [batch_size, 1, 1, cfg.num_kps])), [1,2])
        y_out = tf.reduce_sum(tf.multiply(heatmap_outs, tf.tile(tf.reshape(yy,[1, output_shape[0], output_shape[1], 1]), [batch_size, 1, 1, cfg.num_kps])), [1,2])
        coord_out = tf.concat([tf.reshape(x_out, [batch_size, cfg.num_kps, 1]) \
                                  ,tf.reshape(y_out, [batch_size, cfg.num_kps, 1])] \
                              , axis=2)
        coord_out = coord_out - 1

        coord_out = coord_out / output_shape[0] * cfg.input_shape[0]

        return coord_out

    def render_onehot_heatmap(self, coord, output_shape, cfg):
        batch_size = tf.shape(coord)[0]

        #### [batch, 17] -> [batch * 17]
        x = tf.reshape(coord[:,:,0] / cfg.input_shape[1] * output_shape[1],[-1])
        y = tf.reshape(coord[:,:,1] / cfg.input_shape[0] * output_shape[0],[-1])
        x_floor = tf.floor(x)
        y_floor = tf.floor(y)

        #### [17 ,batch] -> [batch, 17] -> [batch * 17, 1] with batch range indices
        indices_batch = tf.expand_dims(tf.to_float( \
            tf.reshape(
                tf.transpose( \
                    tf.tile( \
                        tf.expand_dims(tf.range(batch_size),0) \
                        ,[cfg.num_kps,1]) \
                    ,[1,0]) \
                ,[-1])),1)

        #### [4 * batch * 17, 1]
        indices_batch = tf.concat([indices_batch, indices_batch, indices_batch, indices_batch], axis=0)

        #### [batch * 17, 1] with 17 range indices
        indices_joint = tf.to_float(tf.expand_dims(tf.tile(tf.range(cfg.num_kps),[batch_size]),1))
        #### [4 * batch * 17, 1]
        indices_joint = tf.concat([indices_joint, indices_joint, indices_joint, indices_joint], axis=0)

        #### [batch * 17, 1] -> [batch * 17, 2]
        indices_lt = tf.concat([tf.expand_dims(y_floor,1), tf.expand_dims(x_floor,1)], axis=1)
        indices_lb = tf.concat([tf.expand_dims(y_floor+1,1), tf.expand_dims(x_floor,1)], axis=1)
        indices_rt = tf.concat([tf.expand_dims(y_floor,1), tf.expand_dims(x_floor+1,1)], axis=1)
        indices_rb = tf.concat([tf.expand_dims(y_floor+1,1), tf.expand_dims(x_floor+1,1)], axis=1)

        #### [4 * batch * 17, 2] -> [4 * batch * 17, 4]
        indices = tf.concat([indices_lt, indices_lb, indices_rt, indices_rb], axis=0)
        indices = tf.cast(indices, tf.float32)
        indices = tf.cast(tf.concat([indices_batch, indices, indices_joint], axis=1),tf.int32)

        prob_lt = (1 - (x - x_floor)) * (1 - (y - y_floor))
        prob_lb = (1 - (x - x_floor)) * (y - y_floor)
        prob_rt = (x - x_floor) * (1 - (y - y_floor))
        prob_rb = (x - x_floor) * (y - y_floor)
        probs = tf.concat([prob_lt, prob_lb, prob_rt, prob_rb], axis=0)

        heatmap = tf.scatter_nd(indices, probs, (batch_size, *output_shape, cfg.num_kps))

        normalizer = tf.reshape(tf.reduce_sum(heatmap,axis=[1,2]),[batch_size,1,1,cfg.num_kps])
        normalizer = tf.where(tf.equal(normalizer,0),tf.ones_like(normalizer),normalizer)
        heatmap = heatmap / normalizer

        return heatmap

    def diy_loss_func(self, cfg, stage_heatmap, gt_C, mask):
        heatmap_outs = stage_heatmap
        target_valid = mask

        target_coord = tf.cast(gt_C * 8, dtype=tf.int32)
        gt_heatmap = tf.stop_gradient(tf.reshape(tf.transpose( \
            self.render_onehot_heatmap(target_coord, cfg.output_shape, cfg), \
            [0, 3, 1, 2]), [cfg.batch_size, cfg.num_kps, -1]))

        gt_coord = target_coord / cfg.input_shape[0] * cfg.output_shape[0]

        # heatmap loss
        out = tf.reshape(tf.transpose(heatmap_outs, [0, 3, 1, 2]), [cfg.batch_size, cfg.num_kps, -1])
        gt = gt_heatmap
        valid_mask = tf.reshape(target_valid, [cfg.batch_size, cfg.num_kps])
        valid_mask = tf.cast(valid_mask, tf.float32)
        loss_heatmap = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=gt, logits=out) * valid_mask)

        # coordinate loss
        out = self.extract_coordinate(heatmap_outs, cfg) / cfg.input_shape[0] * cfg.output_shape[0]
        gt = gt_coord
        valid_mask = tf.reshape(target_valid, [cfg.batch_size, cfg.num_kps, 1])
        valid_mask = tf.cast(valid_mask, tf.float32)

        gt = tf.cast(gt, tf.float32)
        loss_coord = tf.reduce_mean(tf.abs(out - gt) * valid_mask)

        loss = loss_heatmap + loss_coord
        return loss

    def build_loss(self, gt_heatmap, gt_C, mask, cfg):
        '''
        if int(gt_heatmap.get_shape()[1]) != 32:
            gt_heatmap = tf.image.resize_images(gt_heatmap, size=(32, 32))
        '''
        assert cfg.use_loss in ["l2", "refine"]

        self.gt_heatmap = gt_heatmap
        self.total_loss = 0

        for stage in range(self.stages):
            with tf.variable_scope('stage' + str(stage + 1) + '_loss'):
                if cfg.use_loss == "l2":
                    self.stage_loss[stage] = tf.nn.l2_loss(self.stage_heatmap[stage] - self.gt_heatmap,
                                                           name='l2_loss') / self.batch_size
                else:
                    self.stage_loss[stage] = self.diy_loss_func(cfg ,self.stage_heatmap[stage],
                                                                gt_C, mask)
            tf.summary.scalar('stage' + str(stage + 1) + '_loss', self.stage_loss[stage])

        with tf.variable_scope('total_loss'):
            for stage in range(self.stages):
                self.total_loss += self.stage_loss[stage]
            tf.summary.scalar('total loss', self.total_loss)

        self.merged_summary = tf.summary.merge_all()

    def load_weights_from_file(self, weight_file_path, sess, finetune=True):
        weights = pickle.load(open(weight_file_path, 'rb'), encoding='latin1')

        with tf.variable_scope('', reuse=True):
            ## Pre stage conv
            # conv1
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer) + '/bias')

                loaded_kernel = weights['conv1_' + str(layer)]
                loaded_bias = weights['conv1_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv2
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 2) + '/bias')

                loaded_kernel = weights['conv2_' + str(layer)]
                loaded_bias = weights['conv2_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv3
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 4) + '/bias')

                loaded_kernel = weights['conv3_' + str(layer)]
                loaded_bias = weights['conv3_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv4
            for layer in range(1, 3):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 8) + '/bias')

                loaded_kernel = weights['conv4_' + str(layer)]
                loaded_bias = weights['conv4_' + str(layer) + '_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv4_CPM
            for layer in range(1, 5):
                conv_kernel = tf.get_variable('sub_stages/sub_conv' + str(layer + 10) + '/kernel')
                conv_bias = tf.get_variable('sub_stages/sub_conv' + str(layer + 10) + '/bias')

                loaded_kernel = weights['conv4_' + str(2 + layer) + '_CPM']
                loaded_bias = weights['conv4_' + str(2 + layer) + '_CPM_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

            # conv5_3_CPM
            conv_kernel = tf.get_variable('sub_stages/sub_stage_img_feature/kernel')
            conv_bias = tf.get_variable('sub_stages/sub_stage_img_feature/bias')

            loaded_kernel = weights['conv4_7_CPM']
            loaded_bias = weights['conv4_7_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            ## stage 1
            conv_kernel = tf.get_variable('stage_1/conv1/kernel')
            conv_bias = tf.get_variable('stage_1/conv1/bias')

            loaded_kernel = weights['conv5_1_CPM']
            loaded_bias = weights['conv5_1_CPM_b']

            sess.run(tf.assign(conv_kernel, loaded_kernel))
            sess.run(tf.assign(conv_bias, loaded_bias))

            if finetune != True:
                conv_kernel = tf.get_variable('stage_1/stage_heatmap/kernel')
                conv_bias = tf.get_variable('stage_1/stage_heatmap/bias')

                loaded_kernel = weights['conv5_2_CPM']
                loaded_bias = weights['conv5_2_CPM_b']

                sess.run(tf.assign(conv_kernel, loaded_kernel))
                sess.run(tf.assign(conv_bias, loaded_bias))

                ## stage 2 and behind
                for stage in range(2, self.stages + 1):
                    for layer in range(1, 8):
                        conv_kernel = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/kernel')
                        conv_bias = tf.get_variable('stage_' + str(stage) + '/mid_conv' + str(layer) + '/bias')

                        loaded_kernel = weights['Mconv' + str(layer) + '_stage' + str(stage)]
                        loaded_bias = weights['Mconv' + str(layer) + '_stage' + str(stage) + '_b']

                        sess.run(tf.assign(conv_kernel, loaded_kernel))
                        sess.run(tf.assign(conv_bias, loaded_bias))


def cpm_construct(cfg):
    batch_size = cfg.batch_size

    input_image = tf.placeholder(dtype=tf.float32, shape=[batch_size, 256, 256, 3 + 17])
    gt_heatmap = tf.placeholder(dtype=tf.float32, shape=[batch_size, 32 * 1, 32 * 1, 17])
    gt_C = tf.placeholder(dtype=tf.float32, shape=[batch_size, 17, 2])
    mask = tf.placeholder(dtype=tf.int32, shape=[batch_size, 17])

    cpm_m = CPM_Model(stages = 3, joints = 17)
    cpm_m.build_model(
        input_image=input_image, center_map=tf.zeros(shape=[batch_size, 256, 256, 1]),
        batch_size=batch_size
    )
    cpm_m.build_loss(gt_heatmap, gt_C, mask, cfg)
    req_output = cpm_m.stage_heatmap[-1]

    return (input_image, gt_heatmap, cpm_m, req_output, gt_C, mask)


if __name__ == "__main__":

    pass