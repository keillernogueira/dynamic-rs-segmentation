import math
import random
import sys
import datetime
from os import listdir

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score


NUM_CLASSES = 2  # coffee and non coffee


class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def softmax(array):
    return np.exp(array) / np.sum(np.exp(array), axis=0)


def print_params(list_params):
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for i in xrange(1, len(sys.argv)):
        print list_params[i - 1] + '= ' + sys.argv[i]
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


def select_batch(shuffle, batch_size, it, total_size):
    batch = shuffle[it:min(it + batch_size, total_size)]
    if min(it + batch_size, total_size) == total_size or total_size == it + batch_size:
        shuffle = np.asarray(random.sample(xrange(total_size), total_size))
        # print "in", shuffle
        it = 0
        if len(batch) < batch_size:
            diff = batch_size - len(batch)
            batch_c = shuffle[it:it + diff]
            batch = np.concatenate((batch, batch_c))
            it = diff
            # print 'c', batch_c, batch, it
    else:
        it += batch_size
    return shuffle, batch, it


def define_multinomial_probs(values, diff_prob=2):
    interval_size = values[-1] - values[0] + 1

    general_prob = 1.0 / float(interval_size)
    max_prob = general_prob * diff_prob  # for values

    probs = np.full(interval_size, (1.0 - max_prob * len(values)) / float(interval_size - len(values)))
    for i in xrange(len(values)):
        probs[values[i] - values[0]] = max_prob

    return probs


def normalize_images(data, mean_full, std_full):
    # print(training_data.shape)
    # print(test_data.shape)

    data[:, :, :, 0] = np.subtract(data[:, :, :, 0], mean_full[0])
    data[:, :, :, 1] = np.subtract(data[:, :, :, 1], mean_full[1])
    data[:, :, :, 2] = np.subtract(data[:, :, :, 2], mean_full[2])

    data[:, :, :, 0] = np.divide(data[:, :, :, 0], std_full[0])
    data[:, :, :, 1] = np.divide(data[:, :, :, 1], std_full[1])
    data[:, :, :, 2] = np.divide(data[:, :, :, 2], std_full[2])


# for i in range(len(data)):
# data[i] = np.subtract(data[i], mean_full)
# data[i] = np.divide(data[i], std_full)

# print(training_data[0][0])

def compute_image_mean(data):
    mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    std_full = np.std(data, axis=0, ddof=1)[0, 0, :]

    return mean_full, std_full


def load_imgs_torch(files):
    img_data = np.empty([len(files) / 2, 500, 500, 3], dtype=np.float32)
    mask_data = np.empty([len(files) / 2, 500, 500, 1], dtype=np.float32)

    count_files = 0
    mark_files = 0
    for f in files:
        # print f
        i = 0
        try:
            file_in = open(f)
        except IOError:
            print BatchColors.FAIL + "Could not open file " + f + BatchColors.ENDC

        for line in file_in:
            if i == 7:
                c, h, w = line.split(' ')
            if i <= 16:
                i = i + 1
                continue
            arr_chw = np.reshape(np.asarray(line.split(' '), dtype=np.float32), (int(c), int(h), int(w)))
            arr_hcw = np.swapaxes(arr_chw, 0, 1)
            arr_hwc = np.swapaxes(arr_hcw, 1, 2)
            if mark_files % 2 == 0:
                img_data[count_files, :, :, :] = arr_hwc
            else:
                mask_data[count_files, :, :] = np.floor(arr_hwc + 0.5)
                # print f
                # print np.floor(arr_hwc+0.5).astype(int).flatten().shape
                # print np.bincount(np.floor(arr_hwc+0.5).astype(int).flatten())
                count_files = count_files + 1
        mark_files = mark_files + 1

    return img_data, mask_data


def load_images_torch(path):
    files = []
    # masks = []

    for f in listdir(path):
        if "txt" in f and (f != "Thumbs.db" and "jpeg" not in f):
            files.append(path + f)

    files = sorted(files, key=str.lower)

    return load_imgs_torch(files)


def create_crops(data, mask_data, crop_size, is_train, data_aug=True, data_aug_exp=False):
    crops_data = []
    crops_class = []
    pos = []

    for i in xrange(len(data)):
        for j in xrange(0, len(data[i]) - 1, crop_size):
            for k in range(0, len(data[i][j]) - 1, crop_size):
                # print j, k

                # Crop
                crop = data[i, j:j + crop_size, k:k + crop_size, :]
                if len(crop) != crop_size or len(crop[0]) != crop_size:
                    print crop.size
                crops_data.append(crop)

                # Class
                current_class = mask_data[i, j:j + crop_size, k:k + crop_size]
                crops_class.append(current_class)

                if is_train is True and data_aug is True:  # train
                    # LEFT <-> RIGHT
                    # CROP
                    mirrored_lf_crop = np.fliplr(crop)
                    crops_data.append(mirrored_lf_crop)
                    # class
                    mirrored_class = np.fliplr(current_class)
                    crops_class.append(mirrored_class)

                    if data_aug_exp is True:
                        # TOP <-> DOWN
                        # CROP
                        mirrored_ud_crop = np.flipud(crop)
                        crops_data.append(mirrored_ud_crop)
                        # class
                        mirrored_class = np.flipud(current_class)
                        crops_class.append(mirrored_class)

                if is_train is False:  # test
                    current_pos = np.zeros(2)
                    current_pos[0] = int(j)
                    current_pos[1] = int(k)
                    pos.append(current_pos)

    # print len(crops_data)
    # print len(crops_class)
    if is_train is True:
        return np.asarray(crops_data), np.asarray(crops_class, dtype=np.int32)
    else:
        return np.asarray(crops_data), np.asarray(crops_class, dtype=np.int32), pos


def create_crops_stride(data, mask_data, crop_size, stride=1, is_train=False, data_aug=True, data_aug_exp=True):
    crops_data = []
    crops_class = []
    pos = []

    for i in xrange(len(data)):
        j = 0
        count_x = 0
        while j < len(data[i]):  # for j in xrange(0,len(data[i]),stride):
            k = 0
            count_y = 0
            while k < len(data[i][j]):  # for k in xrange(0,len(data[i][j]),stride):

                if j + crop_size <= len(data[i]) and k + crop_size <= len(data[i][j]):
                    # print j, k
                    # print j, j+crop_size
                    # print k, k+crop_size
                    # Crop
                    crop = data[i, j:j + crop_size, k:k + crop_size, :]
                    if len(crop) != crop_size or len(crop[0]) != crop_size:
                        print crop.size
                    crops_data.append(crop)

                    # Class
                    current_class = mask_data[i, j:j + crop_size, k:k + crop_size]
                    crops_class.append(current_class)

                    if is_train is True and data_aug is True:  # train
                        # LEFT <-> RIGHT
                        # CROP
                        mirrored_lf_crop = np.fliplr(crop)
                        crops_data.append(mirrored_lf_crop)
                        # class
                        mirrored_class = np.fliplr(current_class)
                        crops_class.append(mirrored_class)

                        if data_aug_exp is True:
                            # TOP <-> DOWN
                            # CROP
                            mirrored_ud_crop = np.flipud(crop)
                            crops_data.append(mirrored_ud_crop)
                            # class
                            mirrored_class = np.flipud(current_class)
                            crops_class.append(mirrored_class)

                    if is_train is False:  # test
                        current_pos = np.zeros(2)
                        current_pos[0] = int(j)
                        current_pos[1] = int(k)
                        pos.append(current_pos)

                if crop_size % 2 != 0 and count_y % 2 != 0:
                    k += (stride + 1)
                else:
                    k += stride
                count_y += 1

            if crop_size % 2 != 0 and count_x % 2 != 0:
                j += (stride + 1)
            else:
                j += stride
            count_x += 1

    if is_train is True:
        return np.asarray(crops_data), np.asarray(crops_class, dtype=np.int32)
    else:
        return np.asarray(crops_data), np.asarray(crops_class, dtype=np.int32), pos


def dynamically_create_patches(data, mask_data, crop_size, class_distribution, shuffle):
    patches = []
    classes = []

    for i in shuffle:
        if i >= 2 * len(class_distribution):
            cur_pos = i - 2 * len(class_distribution)
        elif i >= len(class_distribution):
            cur_pos = i - len(class_distribution)
        else:
            cur_pos = i

        cur_map = class_distribution[cur_pos][0]
        cur_x = class_distribution[cur_pos][1][0]
        cur_y = class_distribution[cur_pos][1][1]
        # curTransform = class_distribution[cur_pos][2]
        # print cur_map, cur_x, cur_y, curTransform, mask

        patch = data[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        current_class = mask_data[cur_map, cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

        if len(patch) != crop_size and len(patch[0]) != crop_size:
            patch = data[cur_map, cur_x - (crop_size - len(patch)):cur_x + crop_size,
                    cur_y - (crop_size - len(patch[0])):cur_y + crop_size, :]
            current_class = mask_data[cur_map, cur_x - (crop_size - len(current_class)):cur_x + crop_size,
                            cur_y - (crop_size - len(current_class[0])):cur_y + crop_size]
        elif len(patch) != crop_size:
            patch = data[cur_map, cur_x - (crop_size - len(patch)):cur_x + crop_size, cur_y:cur_y + crop_size, :]
            current_class = mask_data[cur_map, cur_x - (crop_size - len(current_class)):cur_x + crop_size,
                            cur_y:cur_y + crop_size]
        elif len(patch[0]) != crop_size:
            patch = data[cur_map, cur_x:cur_x + crop_size, cur_y - (crop_size - len(patch[0])):cur_y + crop_size, :]
            current_class = mask_data[cur_map, cur_x:cur_x + crop_size,
                            cur_y - (crop_size - len(current_class[0])):cur_y + crop_size]

        if len(patch) != crop_size or len(patch[0]) != crop_size:
            print "Error: Current patch size ", len(patch), len(patch[0])
            print cur_x, (crop_size - len(patch)), cur_x - (crop_size - len(patch)), cur_x + crop_size, cur_y, (
                crop_size - len(patch[0])), cur_y - (crop_size - len(patch[0])), cur_y + crop_size
            return
        if len(current_class) != crop_size or len(current_class[0]) != crop_size:
            print "Error: Current class size ", len(current_class), len(current_class[0])
            return

        if i < len(class_distribution):
            patches.append(patch)
            classes.append(current_class)
        elif i < 2 * len(class_distribution):
            patches.append(np.fliplr(patch))
            classes.append(np.fliplr(current_class))
        elif i >= 2 * len(class_distribution):
            patches.append(np.flipud(patch))
            classes.append(np.flipud(current_class))

    return np.asarray(patches, dtype=np.float16), np.asarray(classes, dtype=np.int8)


def create_patches_per_map(data, mask_data, crop_size, stride_crop, index, batch_size):
    patches = []
    classes = []
    pos = []

    h, w, c = data.shape
    # h_m, w_m, c_m = mask_data.shape
    total_index = (int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
        ((h - crop_size) / stride_crop)) + 2)

    count = 0
    # offset_h = int(((index*batch_size*stride_crop)/((w-crop_size)))*stride_crop)
    # offset_h = int((index*batch_size)/(((h-crop_size)/stride_crop)+1))*stride_crop
    # offset_w = int((index*batch_size)%(((w-crop_size)/stride_crop)+1))*stride_crop

    offset_h = int((index * batch_size) / total_index) * stride_crop
    offset_w = int((index * batch_size) % total_index) * stride_crop
    first = True

    # print offset_h, offset_w, h-crop_size, (h-crop_size+1 if ((h-crop_size)%stride_crop) == 0 else h-crop_size+2)

    for j in xrange(offset_h, total_index * stride_crop, stride_crop):
        if first is False:
            offset_w = 0
        for k in xrange(offset_w, total_index * stride_crop, stride_crop):
            if first is True:
                first = False
            cur_x = j
            cur_y = k

            # print cur_x, cur_y, cur_x+crop_size, cur_y+crop_size, data[cur_x:cur_x+crop_size, cur_y:cur_y+crop_size,:]
            patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

            if len(patch) != crop_size and len(patch[0]) != crop_size:
                cur_x = cur_x - (crop_size - len(patch))
                cur_y = cur_y - (crop_size - len(patch[0]))
                patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            elif len(patch) != crop_size:
                cur_x = cur_x - (crop_size - len(patch))
                patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
            elif len(patch[0]) != crop_size:
                cur_y = cur_y - (crop_size - len(patch[0]))
                patch = data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

            if len(patch) != crop_size or len(patch[0]) != crop_size:
                print "Error: Current patch size ", len(patch), len(patch[0])

            count += 1
            patches.append(patch)
            cur_mask_patch = mask_data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
            classes.append(cur_mask_patch)
            # print cur_x, cur_y, cur_x+crop_size, cur_y+crop_size, patch.shape, cur_mask_patch.shape

            current_pos = np.zeros(2)
            current_pos[0] = int(cur_x)
            current_pos[1] = int(cur_y)
            pos.append(current_pos)

            if count == batch_size:  # when completes the batch
                # print "--------- batch complete"
                return np.asarray(patches), np.asarray(classes, dtype=np.int32), pos

    # when its not the total size of the batch
    # print "--------- end without batch complete"
    return np.asarray(patches), np.asarray(classes, dtype=np.int32), pos


def create_mean_and_std(training_data, training_mask_data, crop_size, stride_crop):
    training_patches, _ = create_crops_stride(training_data, training_mask_data, crop_size, stride=stride_crop,
                                              is_train=True)
    return compute_image_mean(training_patches)


def create_distributions_over_classes(labels, crop_size, stride_crop):
    classes = [[[] for i in range(0)] for i in range(NUM_CLASSES)]

    for k in xrange(len(labels)):
        w, h, c = labels[k].shape

        for i in xrange(0, w, stride_crop):
            for j in xrange(0, h, stride_crop):
                patch_class = np.squeeze(labels[k][i:i + crop_size, j:j + crop_size])

                if patch_class.shape == (crop_size, crop_size):
                    count = np.bincount(patch_class.astype(int).flatten())
                    classes[int(np.argmax(count))].append((k, (i, j)))

    return classes[0] + classes[1]


def create_prediction_map(path, all_predcs, pos, step, crop_size):
    im_array = np.empty([500, 500], dtype=np.uint8)

    for i in range(len(all_predcs)):
        im_array[pos[i][1]:pos[i][1] + crop_size, pos[i][0]:pos[i][0] + crop_size] = all_predcs[i, :, :]

    # print np.bincount(im_array.flatten())

    # scipy.misc.imsave(path + 'fcn/tensorflow_predMap_step' + str(step) + '.jpeg', im_array)
    img = Image.fromarray(np.uint8(im_array * 255))
    img.save(path + 'predMap_step' + str(step) + '.jpeg')


def save_map(path, step, prob_im_argmax):
    img = Image.fromarray(np.uint8(prob_im_argmax * 255))
    img.save(path + 'predMap_step' + str(step) + '.jpeg')


def calc_accuracy_by_crop(true_crop, pred_crop, track_conf_matrix):
    b, h, w = pred_crop.shape
    _trueCrop = np.reshape(true_crop, (b, h, w))

    acc = 0
    local_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    # count = 0
    for i in xrange(b):
        for j in xrange(h):
            for k in xrange(w):
                # count += 1
                if _trueCrop[i][j][k] == pred_crop[i][j][k]:
                    acc = acc + 1
                track_conf_matrix[_trueCrop[i][j][k]][pred_crop[i][j][k]] += 1
                local_conf_matrix[_trueCrop[i][j][k]][pred_crop[i][j][k]] += 1

    # print 'count', count
    return acc, local_conf_matrix


def calc_accuracy_by_map(test_mask_data, prob_im_argmax):
    b, h, w, arg = test_mask_data.shape
    acc = 0
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

    for i in xrange(b):
        for j in xrange(h):
            for k in xrange(w):
                # count += 1
                if test_mask_data[i][j][k][0] == prob_im_argmax[j][k]:
                    acc = acc + 1
                conf_matrix[test_mask_data[i][j][k][0]][prob_im_argmax[j][k]] += 1

    # print 'count', count
    return acc, conf_matrix


def select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, is_loss_or_acc='acc',
                           patch_chosen_values=None,
                           debug=False):
    # if 0 in patch_occur:
    patch_occur[np.where(patch_occur == 0)] = 1
    patch_mean = patch_acc_loss / patch_occur
    # print is_loss_or_acc

    if is_loss_or_acc == 'acc':
        argmax_acc = np.argmax(softmax(patch_mean))
        if distribution_type == 'multi_fixed':
            cur_val = int(values[argmax_acc])
        elif distribution_type == 'uniform' or distribution_type == 'multinomial':
            cur_val = values[0] + argmax_acc

        if patch_chosen_values is not None:
            patch_chosen_values[int(argmax_acc)] += 1

        if debug is True:
            print 'errorLoss', patch_acc_loss
            print 'patch_occur', patch_occur
            print 'patch_mean', patch_mean
            print 'argmax_acc', argmax_acc

            print 'specific', argmax_acc, patch_acc_loss[argmax_acc], patch_occur[argmax_acc], patch_mean[argmax_acc]

    elif is_loss_or_acc == 'loss':
        arg_sort_out = np.argsort(patch_mean)

        if debug is True:
            print 'errorLoss', patch_acc_loss
            print 'patch_occur', patch_occur
            print 'patch_mean', patch_mean
            print 'arg_sort_out', arg_sort_out
        if distribution_type == 'multi_fixed':
            for i in xrange(len(values)):
                if patch_occur[arg_sort_out[i]] > 0:
                    cur_val = int(values[arg_sort_out[i]])  # -1*(i+1)
                    if patch_chosen_values is not None:
                        patch_chosen_values[arg_sort_out[i]] += 1
                    if debug is True:
                        print 'specific', arg_sort_out[i], patch_acc_loss[arg_sort_out[i]], \
                            patch_occur[arg_sort_out[i]], patch_mean[arg_sort_out[i]]
                    break
        elif distribution_type == 'uniform' or distribution_type == 'multinomial':
            for i in xrange(values[-1] - values[0] + 1):
                if patch_occur[arg_sort_out[i]] > 0:
                    cur_val = values[0] + arg_sort_out[i]
                    if patch_chosen_values is not None:
                        patch_chosen_values[arg_sort_out[i]] += 1
                    if debug is True:
                        print 'specific', arg_sort_out[i], patch_acc_loss[arg_sort_out[i]], \
                            patch_occur[arg_sort_out[i]], patch_mean[arg_sort_out[i]]
                    break

    if debug is True:
        print 'Current patch size ', cur_val
        if patch_chosen_values is not None:
            print 'Distr of chosen sizes ', patch_chosen_values

    return cur_val


'''
TensorFlow
'''


def leaky_relu(x, alpha=0.1):
    return tf.maximum(alpha * x, x)


def identity_initializer(scale=1.0):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        if len(shape) != 2 or shape[0] != shape[1]:
            raise ValueError('Identity matrix initializer can only be used for 2D square matrices.')
        else:
            return tf.constant_op.constant(scale * np.identity(shape[0], dtype), dtype=dtype)

    return _initializer


def _variable_on_cpu(name, shape, ini):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=ini, dtype=tf.float32)
    return var


def _variable_with_weight_decay(name, shape, ini, weight_decay):
    var = _variable_on_cpu(name, shape, ini)
    # tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)
    # tf.contrib.layers.xavier_initializer(dtype=tf.float32))
    # tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))
    # orthogonal_initializer()
    if weight_decay is not None:
        try:
            weight_decay = tf.mul(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        except:
            weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    return var


def _batch_norm(input_data, is_training, scope=None):
    # Note: is_training is tf.placeholder(tf.bool) type
    return tf.cond(is_training,
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=True, center=False,
                                                        updates_collections=None,
                                                        scope=scope),
                   lambda: tf.contrib.layers.batch_norm(input_data, is_training=False, center=False,
                                                        updates_collections=None, scope=scope, reuse=True)
                   )


def _fc_layer(input, layer_shape, weight_decay, name, activation=None):
    with tf.variable_scope(name):
        weights = _variable_with_weight_decay('weights', shape=layer_shape,
                                              ini=tf.truncated_normal_initializer(stddev=0.005, dtype=tf.float32),
                                              weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', layer_shape[-1], tf.constant_initializer(0.1))

        fc = tf.matmul(input, weights)
        fc = tf.add(fc, biases)

        if activation == 'relu':
            fc = tf.nn.relu(fc, name=name)

        return fc


def _squeeze_excitation_layer(input_data, weight_decay, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = tf.reduce_mean(input_data, axis=[1,2])
        out_dim = squeeze.get_shape().as_list()[1]
        # print(squeeze.get_shape().as_list(), out_dim)

        excitation = _fc_layer(squeeze, [out_dim, out_dim/ratio], weight_decay, layer_name+'_fc1')
        excitation = tf.nn.relu(excitation)
        excitation = _fc_layer(excitation, [out_dim/ratio, out_dim], weight_decay, name=layer_name+'_fc2')
        excitation = tf.nn.sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

        scale = input_data * excitation

    return scale


def _conv_layer(input_data, layer_shape, name, weight_decay, is_training, rate=1, strides=None, pad='SAME',
                activation='relu', batch_norm=True, has_activation=True, is_normal_conv=False,
                init_func=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32)):
    if strides is None:
        strides = [1, 1, 1, 1]
    with tf.variable_scope(name) as scope:
        weights = _variable_with_weight_decay('weights', shape=layer_shape, ini=init_func, weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', layer_shape[-1], tf.constant_initializer(0.1))

        if is_normal_conv is False:
            conv_op = tf.nn.atrous_conv2d(input_data, weights, rate=rate, padding=pad)
        else:
            conv_op = tf.nn.conv2d(input_data, weights, strides=strides, padding=pad)
        conv_act = tf.nn.bias_add(conv_op, biases)

        if batch_norm is True:
            conv_act = _batch_norm(conv_act, is_training, scope=scope)
        if has_activation is True:
            if activation == 'relu':
                conv_act = tf.nn.relu(conv_act, name=name)
            else:
                conv_act = leaky_relu(conv_act)

        return conv_act


def _squeeze_conv_layer(input_data, in_dim, out_dim, k_dim, kernel_size, name, weight_decay, is_training, rate=1,
                        strides=None, pad='SAME', activation='relu', batch_norm=True, has_activation=True,
                        is_normal_conv=False):
    conv1 = _conv_layer(input_data, [1, 1, in_dim, k_dim], name + '_s1', weight_decay, is_training, rate, strides, pad,
                        activation, batch_norm, has_activation, is_normal_conv)

    conv2_1 = _conv_layer(conv1, [1, 1, k_dim, out_dim/2], name + '_s2_1', weight_decay, is_training, rate, strides,
                          pad, activation, batch_norm, has_activation, is_normal_conv)
    conv2_2 = _conv_layer(conv1, [kernel_size, kernel_size, k_dim, out_dim/2], name + '_s2_2', weight_decay,
                          is_training, rate, strides, pad, activation, batch_norm, has_activation, is_normal_conv)

    try:
        out = tf.concat([conv2_1, conv2_2], 3)
    except:
        out = tf.concat(concat_dim=3, values=[conv2_1, conv2_2])

    return out


def _max_pool(input_data, kernel, strides, name, pad='SAME', debug=False):
    pool = tf.nn.max_pool(input_data, ksize=kernel, strides=strides, padding=pad, name=name)
    if debug:
        pool = tf.Print(pool, [tf.shape(pool)], message='Shape of %s' % name)

    return pool


def _avg_pool(input_data, kernel, strides, name, pad='SAME', debug=False):
    pool = tf.nn.avg_pool(input_data, ksize=kernel, strides=strides, padding=pad, name=name)
    if debug:
        pool = tf.Print(pool, [tf.shape(pool)], message='Shape of %s' % name)

    return pool


def dilated_icpr_original(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25
    # print x.get_shape()

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1)
    # pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=1)
    # pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], 'conv3', weight_decay, is_training, rate=2)
    # pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=2)
    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=4)
    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=4)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_small(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1)

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2)

    conv3 = _conv_layer(conv2, [4, 4, 64, 64], "conv3", weight_decay, is_training, rate=3)

    conv4 = _conv_layer(conv3, [4, 4, 64, 128], "conv4", weight_decay, is_training, rate=4)

    conv5 = _conv_layer(conv4, [3, 3, 128, 128], "conv5", weight_decay, is_training, rate=5)

    conv6 = _conv_layer(conv5, [3, 3, 128, 128], "conv6", weight_decay, is_training, rate=6)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 128, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1)

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2)

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], "conv3", weight_decay, is_training, rate=3)

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4)

    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=5)

    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=6)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_avgpool(x, dropout, is_training, weight_decay, crop_size, batch_norm=True):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1)
    pool1 = _avg_pool(conv1, kernel=[1, 5, 5, 1], strides=[1, 1, 1, 1], name='pool1')

    conv2 = _conv_layer(pool1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2)
    pool2 = _avg_pool(conv2, kernel=[1, 5, 5, 1], strides=[1, 1, 1, 1], name='pool2')

    conv3 = _conv_layer(pool2, [4, 4, 64, 128], "conv3", weight_decay, is_training, rate=3)
    pool3 = _avg_pool(conv3, kernel=[1, 5, 5, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(pool3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4)
    pool4 = _avg_pool(conv4, kernel=[1, 7, 7, 1], strides=[1, 1, 1, 1], name='pool4')

    conv5 = _conv_layer(pool4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=5)
    pool5 = _avg_pool(conv5, kernel=[1, 7, 7, 1], strides=[1, 1, 1, 1], name='pool5')

    conv6 = _conv_layer(pool5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=6)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_nodilation(x, dropout, is_training, weight_decay, crop_size, batch_norm=True):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training,
                        batch_norm=batch_norm, is_normal_conv=True)

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training,
                        batch_norm=batch_norm, is_normal_conv=True)

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], "conv3", weight_decay, is_training,
                        batch_norm=batch_norm, is_normal_conv=True)

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training,
                        batch_norm=batch_norm, is_normal_conv=True)

    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "conv5", weight_decay, is_training,
                        batch_norm=batch_norm, is_normal_conv=True)

    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training,
                        batch_norm=batch_norm, is_normal_conv=True)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate1(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1)

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=1)

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], "conv3", weight_decay, is_training, rate=1)

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=1)

    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=1)

    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=1)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_vary_rate(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1)

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2)

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], "conv3", weight_decay, is_training, rate=4)

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=1)

    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=2)

    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=4)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_densely(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 32], "conv1", weight_decay, is_training, rate=1)

    conv2 = _conv_layer(conv1, [5, 5, 32, 32], 'conv2', weight_decay, is_training, rate=2)
    try:
        c1 = tf.concat([conv1, conv2], 3)  # c1 = 32+32 = 64
    except:
        c1 = tf.concat(concat_dim=3, values=[conv1, conv2])

    conv3 = _conv_layer(c1, [4, 4, 64, 64], "conv3", weight_decay, is_training, rate=3)
    try:
        c2 = tf.concat([c1, conv3], 3)  # c2 = 64+64 = 128
    except:
        c2 = tf.concat(concat_dim=3, values=[c1, conv3])

    conv4 = _conv_layer(c2, [4, 4, 128, 64], "conv4", weight_decay, is_training, rate=4)
    try:
        c3 = tf.concat([c2, conv4], 3)  # c3 = 128+64 = 192
    except:
        c3 = tf.concat(concat_dim=3, values=[c2, conv4])

    conv5 = _conv_layer(c3, [3, 3, 192, 128], "conv5", weight_decay, is_training, rate=5)
    try:
        c4 = tf.concat([c3, conv5], 3)  # c4 = 192+128 = 320
    except:
        c4 = tf.concat(concat_dim=3, values=[c3, conv5])

    conv6 = _conv_layer(c4, [3, 3, 320, 128], "conv6", weight_decay, is_training, rate=6)
    try:
        c5 = tf.concat([c4, conv6], 3)  # c5 = 320+256 = 448
    except:
        c5 = tf.concat(concat_dim=3, values=[c4, conv6])

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 448, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(c5, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_grsl(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1, activation='lrelu')
    pool1 = _max_pool(conv1, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool1')

    conv2 = _conv_layer(pool1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2, activation='lrelu')
    pool2 = _max_pool(conv2, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool2')

    conv3 = _conv_layer(pool2, [4, 4, 64, 128], 'conv3', weight_decay, is_training, rate=3, activation='lrelu')
    pool3 = _max_pool(conv3, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(pool3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4, activation='lrelu')
    pool4 = _max_pool(conv4, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool4')

    conv5 = _conv_layer(pool4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=5, activation='lrelu')
    pool5 = _max_pool(conv5, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool5')

    conv6 = _conv_layer(pool5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=6, activation='lrelu')
    pool6 = _max_pool(conv6, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool6')

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(pool6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_grsl_rate8(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1, activation='lrelu')
    pool1 = _max_pool(conv1, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool1')

    conv2 = _conv_layer(pool1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2, activation='lrelu')
    pool2 = _max_pool(conv2, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool2')

    conv3 = _conv_layer(pool2, [4, 4, 64, 128], 'conv3', weight_decay, is_training, rate=3, activation='lrelu')
    pool3 = _max_pool(conv3, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(pool3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4, activation='lrelu')
    pool4 = _max_pool(conv4, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool4')

    conv5 = _conv_layer(pool4, [3, 3, 128, 192], "conv5", weight_decay, is_training, rate=5, activation='lrelu')
    pool5 = _max_pool(conv5, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool5')

    conv6 = _conv_layer(pool5, [3, 3, 192, 192], "conv6", weight_decay, is_training, rate=6, activation='lrelu')
    pool6 = _max_pool(conv6, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool6')

    conv7 = _conv_layer(pool6, [3, 3, 192, 256], "conv7", weight_decay, is_training, rate=7, activation='lrelu')
    pool7 = _max_pool(conv7, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool7')

    conv8 = _conv_layer(pool7, [3, 3, 256, 256], "conv8", weight_decay, is_training, rate=8, activation='lrelu')
    pool8 = _max_pool(conv8, kernel=[1, 3, 3, 1], strides=[1, 1, 1, 1], name='pool8')

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_SE(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1)
    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2)
    se1 = _squeeze_excitation_layer(conv2, weight_decay, ratio=4, layer_name='se1')

    conv3 = _conv_layer(se1, [4, 4, 64, 128], "conv3", weight_decay, is_training, rate=3)
    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4)
    se2 = _squeeze_excitation_layer(conv4, weight_decay, ratio=4, layer_name='se2')

    conv5 = _conv_layer(se2, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=5)
    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=6)
    se3 = _squeeze_excitation_layer(conv6, weight_decay, ratio=4, layer_name='se3')

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(se3, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_squeeze(x, dropout, is_training, weight_decay, crop_size):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, 3])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, 3, 64], "conv1", weight_decay, is_training, rate=1)
    conv2 = _squeeze_conv_layer(conv1, 64, 64, 32, 5, 'conv2', weight_decay, is_training, rate=2)

    conv3 = _squeeze_conv_layer(conv2, 64, 128, 64, 4, "conv3", weight_decay, is_training, rate=3)
    conv4 = _squeeze_conv_layer(conv3, 128, 128, 64, 4, "conv4", weight_decay, is_training, rate=4)

    conv5 = _squeeze_conv_layer(conv4, 128, 256, 64, 3, "conv5", weight_decay, is_training, rate=5)
    conv6 = _squeeze_conv_layer(conv5, 256, 256, 128, 3, "conv6", weight_decay, is_training, rate=6)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def loss_def(_logits, _labels):
    logits = tf.reshape(_logits, [-1, NUM_CLASSES])
    labels = tf.cast(tf.reshape(_labels, [-1]), tf.int32)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.add_to_collection('losses', cross_entropy_mean)

    # The total loss is defined as the cross entropy loss plus all of the weight decay terms (L2 loss).
    return tf.add_n(tf.get_collection('losses'), name='total_loss')


def test(sess, test_data, test_mask_data, mean_full, std_full, batch_size, x, y, crop, keep_prob, is_training, pred_up,
         logits, step, crop_size, path):
    stride_crop = int(math.floor(crop_size / 2.0))

    for k in xrange(len(test_data)):
        h, w, c = test_mask_data[k].shape
        # all_size += h*w

        # print (h-crop_size)%stride_crop, (h-crop_size)%stride_crop == 0, int(((h-crop_size)/stride_crop)) + 1,
        #  int(((h-crop_size)/stride_crop)) + 2
        instaces_stride_h = (int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
            ((h - crop_size) / stride_crop)) + 2)
        instaces_stride_w = (int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
            ((w - crop_size) / stride_crop)) + 2)
        # print instaces_stride_h, instaces_stride_w
        instaces_stride = instaces_stride_h * instaces_stride_w

        prob_im = np.zeros([h, w, NUM_CLASSES], dtype=np.float32)
        occur_im = np.zeros([h, w, NUM_CLASSES], dtype=np.uint32)

        for i in xrange(0, (
                    (instaces_stride / batch_size) + 1 if instaces_stride % batch_size != 0
                    else (instaces_stride / batch_size))):
            test_patches, test_classes, pos = create_patches_per_map(test_data[k], test_mask_data[k], crop_size,
                                                                     stride_crop, i, batch_size)
            normalize_images(test_patches, mean_full, std_full)

            bx = np.reshape(test_patches, (-1, crop_size * crop_size * 3))
            by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))

            _pred_up, _logits = sess.run([pred_up, logits],
                                         feed_dict={x: bx, y: by, crop: crop_size, keep_prob: 1., is_training: False})
            for j in xrange(len(_logits)):
                # print pos[j][0], pos[j][0]+crop_size, pos[j][1], pos[j][1]+crop_size
                prob_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
                int(pos[j][1]):int(pos[j][1]) + crop_size,:] += _logits[j, :, :, :]
                occur_im[int(pos[j][0]):int(pos[j][0]) + crop_size, int(pos[j][1]):int(pos[j][1]) + crop_size, :] += 1
                # index += 1

        occur_im[np.where(occur_im == 0)] = 1

        # np.save(output_path + 'prob_map' + str(testingInstances[k]) + '.npy', prob_im/occur_im.astype(float))
        prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=2)
        # print np.bincount(prob_im_argmax.astype(int).flatten())
        save_map(path, step, prob_im_argmax)

        cm_test_per_map = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
        for t in xrange(h):
            for r in xrange(w):
                # print test_mask_data[k][t][r][0], prob_im_argmax[t][r]
                cm_test_per_map[int(test_mask_data[k][t][r][0])][int(prob_im_argmax[t, r])] += 1
                # all_cm_test[test_mask_data[k][t][r]][prob_im_argmax[t][r]] += 1

        _sum = 0.0
        total = 0
        for i in xrange(len(cm_test_per_map)):
            _sum += (
                cm_test_per_map[i][i] / float(np.sum(cm_test_per_map[i])) if np.sum(cm_test_per_map[i]) != 0 else 0)
            total += cm_test_per_map[i][i]

        cur_kappa = cohen_kappa_score(test_mask_data[k].flatten(), prob_im_argmax.flatten())
        cur_f1 = f1_score(test_mask_data[k].flatten(), prob_im_argmax.flatten(), average='macro')

        print("---- Iter " + str(step) + " -- Time " + str(datetime.datetime.now().time()) +
              " -- Test: Overall Accuracy= " + str(total) +
              " Overall Accuracy= " + "{:.6f}".format(total / float(h * w)) +
              " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
              " F1 Score= " + "{:.4f}".format(cur_f1) +
              " Kappa= " + "{:.4f}".format(cur_kappa) +
              " Confusion Matrix= " + np.array_str(cm_test_per_map).replace("\n", "")
              )


'''
python coffee_dilated_random.py /home/coffee/train_txt/ /home/coffee/test_txt/ /home/segmentation_tensorflow/coffee/aux/
0.01 0.001 100 150000 25 5 dilated_icpr
'''


def main():
    list_params = ['path_train', 'path_test', 'output_path(for model, images, etc)', 'currentModelPath', 'learningRate',
                   'weight_decay', 'batch_size', 'niter', 'referenced_crop_size', 'referenced_stride_crop',
                   'net_type[dilated_icpr_original|dilated_grsl|dilated_icpr_rate6|dilated_icpr_rate6_small]',
                   'distribution_type[single_fixed|multi_fixed|uniform|multinomial]', 'probValues',
                   'update_type [acc|loss]']
    if len(sys.argv) < len(list_params) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
    print_params(list_params)

    # training images path
    index = 1
    path_train = sys.argv[index]
    # training images path
    index = index + 1
    path_test = sys.argv[index]
    # test image
    index = index + 1
    output_path = sys.argv[index]
    # current model path
    index = index + 1
    current_model = sys.argv[index]

    # Parameters
    index = index + 1
    lr_initial = float(sys.argv[index])
    index = index + 1
    weight_decay = float(sys.argv[index])
    index = index + 1
    batch_size = int(sys.argv[index])
    index = index + 1
    niter = int(sys.argv[index])
    index = index + 1
    referenced_crop_size = int(sys.argv[index])
    index = index + 1
    referenced_stride_crop = int(sys.argv[index])
    index = index + 1
    net_type = sys.argv[index]

    # distr type
    index = index + 1
    distribution_type = sys.argv[index]
    index = index + 1
    values = [int(i) for i in sys.argv[index].split(',')]
    index = index + 1
    update_type = sys.argv[index]

    display_step = 50

    if distribution_type == 'multi_fixed':
        patch_acc_loss = np.zeros(len(values), dtype=np.float32)
        # errorLoss = np.zeros(len(values), dtype=np.float32)
        patch_occur = np.zeros(len(values), dtype=np.int32)
        patch_chosen_values = np.zeros(len(values), dtype=np.int32)
    elif distribution_type == 'uniform' or distribution_type == 'multinomial':
        patch_acc_loss = np.zeros(values[-1] - values[0] + 1, dtype=np.float32)
        # errorLoss = np.zeros(values[-1]-values[0]+1, dtype=np.float32)
        patch_occur = np.zeros(values[-1] - values[0] + 1, dtype=np.int32)
        patch_chosen_values = np.zeros(values[-1] - values[0] + 1, dtype=np.int32)
        probs = define_multinomial_probs(values)

    # PROCESS IMAGES
    training_data, training_mask_data = load_images_torch(path_train)
    test_data, test_mask_data = load_images_torch(path_test)

    class_distribution = create_distributions_over_classes(training_mask_data, crop_size=referenced_crop_size,
                                                           stride_crop=referenced_stride_crop)
    total_length = len(class_distribution)

    epoch_number = 1000
    val_inteval = 1000

    # create mean, std from training
    mean_full, std_full = create_mean_and_std(training_data, training_mask_data, crop_size=referenced_crop_size,
                                              stride_crop=referenced_stride_crop)

    # TRAIN NETWORK
    # Network Parameters
    # n_input = crop_size*crop_size*3 # RGB
    # n_input_mask = crop_size*crop_size*1 # BW
    dropout = 0.5  # Dropout, probability to keep units

    # tf Graph input_data
    crop = tf.placeholder(tf.int32)
    x = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.float32, [None, None])

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    global_step = tf.Variable(0, name='global_step', trainable=False)

    # CONVNET
    if net_type == 'dilated_icpr_original':
        logits = dilated_icpr_original(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_grsl':
        logits = dilated_grsl(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_icpr_rate6_densely':
        logits = dilated_icpr_rate6_densely(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_grsl_rate8':
        logits = dilated_grsl_rate8(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_icpr_rate6':
        logits = dilated_icpr_rate6(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_icpr_rate6_small':
        logits = dilated_icpr_rate6_small(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_icpr_rate1':
        logits = dilated_icpr_rate1(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_icpr_vary_rate':
        logits = dilated_icpr_vary_rate(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_icpr_rate6_nodilation':
        logits = dilated_icpr_rate6_nodilation(x, keep_prob, is_training, weight_decay, crop, batch_norm=True)
    elif net_type == 'dilated_icpr_rate6_avgpool':
        logits = dilated_icpr_rate6_avgpool(x, keep_prob, is_training, weight_decay, crop, batch_norm=True)
    elif net_type == 'dilated_icpr_rate6_SE':
        logits = dilated_icpr_rate6_SE(x, keep_prob, is_training, weight_decay, crop)
    elif net_type == 'dilated_icpr_rate6_squeeze':
        logits = dilated_icpr_rate6_squeeze(x, keep_prob, is_training, weight_decay, crop)
    else:
        print BatchColors.FAIL + 'Net type not identified: ' + net_type + BatchColors.ENDC
        return

    # Define loss and optimizer
    loss = loss_def(logits, y)

    lr = tf.train.exponential_decay(lr_initial, global_step, 50000, 0.1, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

    # Evaluate model
    pred_up = tf.argmax(logits, dimension=3)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver() # max_to_keep=None)
    # restore
    saver_restore = tf.train.Saver()

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    shuffle = np.asarray(random.sample(xrange(3 * total_length), 3 * total_length))
    current_iter = 1

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    # config=tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session() as sess:
        if 'model' in current_model:
            current_iter = int(current_model.split('-')[-1])
            print BatchColors.OKBLUE + 'Model restored from ' + current_model + BatchColors.ENDC
            patch_acc_loss = np.load(output_path + 'errorAcc_step_' + str(current_iter) + '.npy')
            patch_occur = np.load(output_path + 'errorOccur_step_' + str(current_iter) + '.npy')
            patch_chosen_values = np.load(output_path + 'chosenValues_step_' + str(current_iter) + '.npy')
            # print patch_acc_loss, patch_occur, patch_chosen_values
            saver_restore.restore(sess, current_model)
        else:
            sess.run(init)
            print 'Model totally initialized!'

        # aux variables
        it = 0
        epoch_mean = 0.0
        epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

        # Keep training until reach max iterations
        for step in xrange(current_iter, niter + 1):
            if distribution_type == 'multi_fixed':
                cur_size_int = np.random.randint(len(values))
                cur_batch_size = int(values[cur_size_int])
            elif distribution_type == 'uniform':
                cur_batch_size = int(np.random.uniform(values[0], values[-1] + 1, 1))
                cur_size_int = cur_batch_size - values[0]
            elif distribution_type == 'multinomial':
                cur_size_int = np.random.multinomial(1, probs).argmax()
                cur_batch_size = values[0] + cur_size_int
            elif distribution_type == 'single_fixed':
                cur_batch_size = int(values[0])

            print cur_batch_size # cur_size_int
            # print 'new batch of crop size == ', cur_batch_size
            shuffle, batch, it = select_batch(shuffle, batch_size, it, 3 * total_length)
            if len(batch) != batch_size:
                print BatchColors.FAIL + "Error: size of current batch " + str(len(batch)) + \
                      " differs from batch_size " + str(batch_size) + BatchColors.ENDC
                return
            b_x, b_y = dynamically_create_patches(training_data, training_mask_data, cur_batch_size, class_distribution,
                                                  batch)

            normalize_images(b_x, mean_full, std_full)
            batch_x = np.reshape(b_x, (-1, cur_batch_size * cur_batch_size * 3))
            batch_y = np.reshape(b_y, (-1, cur_batch_size * cur_batch_size * 1))

            # Run optimization op (backprop)
            _, batch_loss, batch_pred_up = sess.run([optimizer, loss, pred_up],
                                                    feed_dict={x: batch_x, y: batch_y, crop: cur_batch_size,
                                                               keep_prob: dropout, is_training: True})

            acc, batch_cm_train = calc_accuracy_by_crop(batch_y, batch_pred_up, epoch_cm_train)
            epoch_mean += acc

            if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                    or distribution_type == 'multinomial':
                # print (batch_loss*(epoch/10.0) if update_type == 'loss' else (acc/float(np.sum(batch_cm_train))))
                patch_acc_loss[cur_size_int] += (
                    batch_loss if update_type == 'loss' else (acc / float(np.sum(batch_cm_train))))  # *(epoch/10.0)
                # errorLoss[cur_size_int] += batch_loss*(epoch/10.0)
                patch_occur[cur_size_int] += 1

            if step != 0 and step % display_step == 0:
                _sum = 0.0
                for i in xrange(len(batch_cm_train)):
                    _sum += (batch_cm_train[i][i] / float(np.sum(batch_cm_train[i]))
                             if np.sum(batch_cm_train[i]) != 0 else 0)

                print("Iter " + str(step) + " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
                      " Absolut Right Pred= " + str(int(acc)) +
                      " Overall Accuracy= " + "{:.4f}".format(acc / float(np.sum(batch_cm_train))) +
                      " Normalized Accuracy= " + "{:.4f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                      )

            if step % epoch_number == 0:
                _sum = 0.0
                for i in xrange(len(epoch_cm_train)):
                    _sum += (epoch_cm_train[i][i] / float(np.sum(epoch_cm_train[i]))
                             if np.sum(epoch_cm_train[i]) != 0 else 0)

                print("-- Iter " + str(step) + " -- Training Epoch:" +
                      " Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(np.sum(epoch_cm_train))) +
                      " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(epoch_cm_train).replace("\n", "")
                      )

                epoch_mean = 0.0
                epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            if step != 0 and step % val_inteval == 0:
                saver.save(sess, output_path + 'model', global_step=step)
                if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                        or distribution_type == 'multinomial':
                    np.save(output_path + 'errorAcc_step_' + str(step) + '.npy', patch_acc_loss)
                    np.save(output_path + 'errorOccur_step_' + str(step) + '.npy', patch_occur)
                    np.save(output_path + 'chosenValues_step_' + str(step) + '.npy', patch_chosen_values)
                # Test: Final
                if distribution_type == 'multi_fixed' or distribution_type == 'uniform' \
                        or distribution_type == 'multinomial':
                    cur_val = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur,
                                                     update_type, patch_chosen_values, debug=True)
                else:
                    cur_val = int(values[0])

                test(sess, test_data, test_mask_data, mean_full, std_full, batch_size, x, y, crop, keep_prob,
                     is_training, pred_up, logits, step, cur_val, output_path)

        print("Optimization Finished!")

        saver.save(sess, output_path + 'model', global_step=step)
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            np.save(output_path + 'errorAcc_step_' + str(step) + '.npy', patch_acc_loss)
            np.save(output_path + 'errorOccur_step_' + str(step) + '.npy', patch_occur)
            np.save(output_path + 'chosenValues_step_' + str(step) + '.npy', patch_chosen_values)
        # Test: Final
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            cur_val = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
                                            patch_chosen_values,
                                            debug=True)
        else:
            cur_val = int(values[0])
        test(sess, test_data, test_mask_data, mean_full, std_full, batch_size, x, y, crop, keep_prob, is_training,
             pred_up, logits, step, cur_val, output_path)


if __name__ == "__main__":
    main()
