import datetime
import math
import os
import random
import sys

import gdal
import numpy as np
import scipy.misc
import tensorflow as tf
from PIL import Image
from skimage import img_as_float
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import f1_score
from scipy.ndimage import rotate


NUM_CLASSES = 6


class BatchColors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def print_params(list_params):
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
    for i in xrange(1, len(sys.argv)):
        print list_params[i - 1] + '= ' + sys.argv[i]
    print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'


def softmax(array):
    expa = np.exp(array)
    sumexp = np.sum(expa, axis=-1)
    sumexp_repeat = np.repeat(sumexp, NUM_CLASSES).reshape((expa.shape))
    soft_calc = np.divide(expa, sumexp_repeat)
    return soft_calc


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


def define_multinomial_probs(values, dif_prob=2):
    interval_size = values[-1] - values[0] + 1

    general_prob = 1.0 / float(interval_size)
    max_prob = general_prob * dif_prob  # for values

    probs = np.full(interval_size, (1.0 - max_prob * len(values)) / float(interval_size - len(values)))
    for i in xrange(len(values)):
        probs[values[i] - values[0]] = max_prob

    return probs


def normalize_images(data, mean_full, std_full):
    data[:, :, :, 0] = np.subtract(data[:, :, :, 0], mean_full[0])
    data[:, :, :, 1] = np.subtract(data[:, :, :, 1], mean_full[1])
    data[:, :, :, 2] = np.subtract(data[:, :, :, 2], mean_full[2])

    data[:, :, :, 0] = np.divide(data[:, :, :, 0], std_full[0])
    data[:, :, :, 1] = np.divide(data[:, :, :, 1], std_full[1])
    data[:, :, :, 2] = np.divide(data[:, :, :, 2], std_full[2])


def compute_image_mean(data):
    mean_full = np.mean(np.mean(np.mean(data, axis=0), axis=0), axis=0)
    std_full = np.std(data, axis=0, ddof=1)[0, 0, :]

    return mean_full, std_full


def retrieve_class_Using_RGB(val):
    # Impervious surfaces (RGB: 255, 255, 255)
    if val[0] == 255 and val[1] == 255 and val[2] == 255:
        current_class = 0
    # Building (RGB: 0, 0, 255)
    elif val[0] == 0 and val[1] == 0 and val[2] == 255:
        current_class = 1
    # Low vegetation (RGB: 0, 255, 255)
    elif val[0] == 0 and val[1] == 255 and val[2] == 255:
        current_class = 2
    # Tree (RGB: 0, 255, 0)
    elif val[0] == 0 and val[1] == 255 and val[2] == 0:
        current_class = 3
    # Car (RGB: 255, 255, 0)
    elif val[0] == 255 and val[1] == 255 and val[2] == 0:
        current_class = 4
    # Clutter/background (RGB: 255, 0, 0)
    elif val[0] == 255 and val[1] == 0 and val[2] == 0:
        current_class = 5

    else:
        print BatchColors.FAIL + "ERROR: Class value not found! " + str(val) + BatchColors.ENDC
        current_class = -1

    return current_class


def retrieve_RGB_using_class(value):
    # Impervious surfaces (RGB: 255, 255, 255)
    if value == 0:
        return (255, 255, 255)
    # Building (RGB: 0, 0, 255)
    elif value == 1:
        return (0, 0, 255)
    # Low vegetation (RGB: 0, 255, 255)
    elif value == 2:
        return (0, 255, 255)
    # Tree (RGB: 0, 255, 0)
    elif value == 3:
        return (0, 255, 0)
    # Car (RGB: 255, 255, 0)
    elif value == 4:
        return (255, 255, 0)
    # Clutter/background (RGB: 255, 0, 0)
    elif value == 5:
        return (255, 0, 0)
    else:
        print BatchColors.FAIL + "ERROR! Class did not find!!! " + str(value) + BatchColors.ENDC
        return 0


def convert_to_class(img_label):
    w, h, c = img_label.shape
    converted_img = np.empty([w, h], dtype=np.uint8)
    for i in xrange(w):
        for j in xrange(h):
            converted_img[i, j] = retrieve_class_Using_RGB(img_label[i, j, :])
    return converted_img


def dynamically_calculate_mean_and_std(data, indexes, crop_size):
    total = indexes[0] + indexes[1] + indexes[2] + indexes[3] + indexes[4] + indexes[5]

    mean_full = []
    std_full = []

    all_patches = []
    # count = 0
    for i in xrange(len(total)):
        cur_map = total[i][0]
        cur_x = total[i][1]
        cur_y = total[i][2]

        patches = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        if len(patches[0]) != crop_size or len(patches[1]) != crop_size:
            print BatchColors.FAIL + "Error! Current patch size: " + str(len(patches)) + "x" + \
                  str(len(patches[0])) + BatchColors.ENDC
            return

        all_patches.append(patches)

        if i > 0 and i % 5000 == 0:
            mean, std = compute_image_mean(np.asarray(all_patches))
            mean_full.append(mean)
            std_full.append(std)
            all_patches = []

    # remaining images
    mean, std = compute_image_mean(np.asarray(all_patches))
    mean_full.append(mean)
    std_full.append(std)

    # print 'count', count
    return np.mean(mean_full, axis=0), np.mean(std_full, axis=0)


def load_images(path, instances, process, image_type='vaihingen'):
    images = []
    masks = []

    for f in instances:
        print BatchColors.OKBLUE + 'Reading instance ' + str(f) + BatchColors.ENDC
        if image_type == 'vaihingen':
            img_ndsm = img_as_float(
                scipy.misc.imread(path + 'normalized_DSM/dsm_09cm_matching_area' + str(f) + '_normalized.jpg'))
            img_ndsm = np.reshape(img_ndsm, (len(img_ndsm), len(img_ndsm[0]), 1))

            img_rgb = img_as_float(scipy.misc.imread(path + 'top/top_mosaic_09cm_area' + str(f) + '.tif'))

            if process == 'validate_test':
                img_label = scipy.misc.imread(path + 'gts_eroded_encoding/top_mosaic_09cm_area' + str(f) +
                                              '_noBoundary.tif')
            elif  process == 'training' or process == 'crf':
                img_label = scipy.misc.imread(path + 'gts_enconding/top_mosaic_09cm_area' + str(f) + '.tif')
            # else:
                # create_final_map
        elif image_type == 'postdam':
            img_ndsm = img_as_float(scipy.misc.imread(path + '1_DSM_normalisation/dsm_potsdam_0' + (
                str(f) if int(f.split("_")[1]) >= 10 else str(f.split("_")[0]) + '_0' + str(
                    f.split("_")[1])) + '_normalized_lastools.jpg'))
            if len(img_ndsm) != len(img_ndsm[0]):
                new_columns = np.zeros([len(img_ndsm), 1], dtype=type(img_ndsm[0, 0]))
                img_ndsm = np.append(img_ndsm, new_columns, axis=1)
            img_ndsm = np.reshape(img_ndsm, (len(img_ndsm), len(img_ndsm[0]), 1))

            # img_rgb = img_as_float(scipy.misc.imread(path+'4_Ortho_RGBIR/top_potsdam_' + str(f) + '_RGBIR.tif'))
            ds = gdal.Open(path + '4_Ortho_RGBIR/top_potsdam_' + str(f) + '_RGBIR.tif')
            img_rgb = np.empty([ds.RasterXSize, ds.RasterYSize, ds.RasterCount], dtype=np.float64)
            for band in xrange(1, ds.RasterCount + 1):
                img_rgb[:, :, band - 1] = img_as_float(np.array(ds.GetRasterBand(band).ReadAsArray()))

            if process == 'validate_test':
                img_label = scipy.misc.imread(path + 'gts_eroded_encoding/top_potsdam_' + str(f) +
                                              '_label_noBoundary.tif')
            elif process == 'training' or process == 'crf':
                img_label = scipy.misc.imread(path + 'gts_enconding/top_potsdam_' + str(f) + '_label.tif')
            # else:
                # create_final_map

        full_img = np.concatenate((img_rgb, img_ndsm), axis=2)

        # if handleBorderType == 'reflect':
        # images.append(manipulateBorderArray(full_img, crop_size))
        # elif handleBorderType == 'zero':
        # images.append(np.lib.pad(full_img, ((mask, mask), (mask, mask), (0, 0)), 'constant', constant_values=0))
        # else:
        images.append(full_img)
        if process == 'validate_test' or process == 'training' or process == 'crf':
            masks.append(img_label)
    # print np.bincount(img_label.flatten())

    return np.asarray(images), np.asarray(masks)


def dynamically_create_patches(data, mask_data, training_instances_batch, crop_size, is_train=True):
    patches = []
    classes = NUM_CLASSES * [0]
    classes_patches = []
    masks = []

    overall_count = 0
    flip_count = 0

    for i in xrange(len(training_instances_batch)):
        cur_map = training_instances_batch[i][0]
        cur_x = training_instances_batch[i][1]
        cur_y = training_instances_batch[i][2]

        cur_patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        if len(cur_patch) != crop_size and len(cur_patch[0]) != crop_size:
            cur_x = cur_x - (crop_size - len(cur_patch))
            cur_y = cur_y - (crop_size - len(cur_patch[0]))
            cur_patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        elif len(cur_patch) != crop_size:
            cur_x = cur_x - (crop_size - len(cur_patch))
            cur_patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]
        elif len(cur_patch[0]) != crop_size:
            cur_y = cur_y - (crop_size - len(cur_patch[0]))
            cur_patch = data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size, :]

        cur_mask_patch = mask_data[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

        if len(cur_patch) != crop_size or len(cur_patch[0]) != crop_size:
            print BatchColors.FAIL + "Error: Current PATCH size is " + str(len(cur_patch)) + "x" + str(
                len(cur_patch[0])) + BatchColors.ENDC
            return
        if len(cur_mask_patch) != crop_size or len(cur_mask_patch[0]) != crop_size:
            print BatchColors.FAIL + "Error: Current MASK size is " + str(len(cur_mask_patch)) + "x" + str(
                len(cur_mask_patch[0])) + BatchColors.ENDC
            return

        cur_class = np.argmax(np.bincount(cur_mask_patch.astype(int).flatten()))
        classes[int(cur_class)] += 1

        cur_mask = np.ones((crop_size, crop_size), dtype=np.bool)

        # DATA AUGMENTATION
        if is_train is True:
            # ROTATION AUGMENTATION
            cur_rot = training_instances_batch[i][3]
            possible_rotation = np.random.randint(0, 2)
            if possible_rotation == 1:  # default = 1
                # print 'rotation'
                cur_patch = scipy.ndimage.rotate(cur_patch, cur_rot, order=0, reshape=False)
                cur_mask_patch = scipy.ndimage.rotate(cur_mask_patch, cur_rot, order=0, reshape=False)
                cur_mask = scipy.ndimage.rotate(cur_mask, cur_rot, order=0, reshape=False)

            # NORMAL NOISE
            possible_noise = np.random.randint(0, 2)
            if possible_noise == 1:
                cur_patch = cur_patch + np.random.normal(0, 0.01, cur_patch.shape)

            # FLIP AUGMENTATION
            possible_noise = np.random.randint(0, 3)
            if possible_noise == 0:
                patches.append(cur_patch)
                classes_patches.append(cur_mask_patch)
                masks.append(cur_mask)
            if possible_noise == 1:
                patches.append(np.flipud(cur_patch))
                classes_patches.append(np.flipud(cur_mask_patch))
                masks.append(np.flipud(cur_mask))
                flip_count += 1
            elif possible_noise == 2:
                patches.append(np.fliplr(cur_patch))
                classes_patches.append(np.fliplr(cur_mask_patch))
                masks.append(np.fliplr(cur_mask))
                flip_count += 1
        else:
            patches.append(cur_patch)
            classes_patches.append(cur_mask_patch)
            masks.append(cur_mask)

        overall_count += 1

    pt_arr = np.asarray(patches)
    cl_arr = np.asarray(classes_patches, dtype=np.int)
    mk_arr = np.asarray(masks, dtype=np.bool)

    # print pt_arr.shape
    # print cl_arr.shape
    # print mk_arr.shape

    return pt_arr, cl_arr, mk_arr


def create_patches_per_map(data, mask_data, crop_size, stride_crop, index, batch_size):
    patches = []
    classes = []
    pos = []

    h, w, c = data.shape
    # h_m, w_m = mask_data.shape
    total_index_h = (int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
        ((h - crop_size) / stride_crop)) + 2)
    total_index_w = (int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
        ((w - crop_size) / stride_crop)) + 2)

    count = 0

    offset_h = int((index * batch_size) / total_index_w) * stride_crop
    offset_w = int((index * batch_size) % total_index_w) * stride_crop
    first = True

    for j in xrange(offset_h, total_index_h * stride_crop, stride_crop):
        if first is False:
            offset_w = 0
        for k in xrange(offset_w, total_index_w * stride_crop, stride_crop):
            if first is True:
                first = False
            cur_x = j
            cur_y = k

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

            count += 1
            cur_mask_patch = mask_data[cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

            if len(patch) != crop_size or len(patch[0]) != crop_size:
                print "Error: Current patch size ", len(patch), len(patch[0])
                return
            if len(cur_mask_patch) != crop_size or len(cur_mask_patch[0]) != crop_size:
                print "Error: Current cur_mask_patch size ", len(cur_mask_patch), len(cur_mask_patch[0])
                return

            patches.append(patch)
            classes.append(cur_mask_patch)
            # print cur_x, cur_y, cur_x+crop_size, cur_y+crop_size, patch.shape, cur_mask_patch.shape

            current_pos = np.zeros(2)
            current_pos[0] = int(cur_x)
            current_pos[1] = int(cur_y)
            pos.append(current_pos)

            if count == batch_size:  # when completes current batch
                # print "--------- batch complete"
                return np.asarray(patches), np.asarray(classes, dtype=np.int8), pos

    # when its not the total size of the batch
    # print "--------- end without batch complete"
    return np.asarray(patches), np.asarray(classes, dtype=np.int8), pos


def select_super_batch_instances(class_distribution, rotation_distribution=None, batch_size=100, super_batch=500):
    instances = []
    overall_count = 0

    samples_per_class = int((batch_size * super_batch) / len(class_distribution))
    # print samples_per_class

    # for each class
    for i in xrange(len(class_distribution)):
        # print len(class_distribution[i]), samples_per_class
        # (samples_per_class if len(class_distribution[i]) >= samples_per_class else len(class_distribution[i]))
        shuffle = np.asarray(random.sample(xrange(len(class_distribution[i])), (
            samples_per_class if len(class_distribution[i]) >= samples_per_class else len(class_distribution[i]))))

        for j in shuffle:
            cur_map = class_distribution[i][j][0]
            cur_x = class_distribution[i][j][1]
            cur_y = class_distribution[i][j][2]
            # print (rotation_distribution[i][j] if (rotation_distribution is not None) else 0)
            cur_rot = (rotation_distribution[i][j] if (rotation_distribution is not None) else 0)

            instances.append((cur_map, cur_x, cur_y, cur_rot))
            overall_count += 1

    # remaining if int((batch_size*superEpoch)/len(class_distribution)) is not divisible
    # print overall_count, (batch_size*super_batch), overall_count != (batch_size*super_batch)
    if overall_count != (batch_size * super_batch):
        lack = (batch_size * super_batch) - overall_count
        # print 'in', lack
        for i in xrange(lack):
            rand_class = np.random.randint(len(class_distribution))
            rand_map = np.random.randint(len(class_distribution[rand_class]))

            cur_map = class_distribution[rand_class][rand_map][0]
            cur_x = class_distribution[rand_class][rand_map][1]
            cur_y = class_distribution[rand_class][rand_map][2]
            cur_rot = (rotation_distribution[rand_class][rand_map] if (rotation_distribution is not None) else 0)

            instances.append((cur_map, cur_x, cur_y, cur_rot))
            overall_count += 1

    # print overall_count, (batch_size*super_batch), overall_count != (batch_size*super_batch)
    assert overall_count == (batch_size * super_batch), "Could not select ALL instances"
    # for i in xrange(len(instances)):
    # print 'Instances ' + str(i)  + ' has length ' + str(len(instances[i]))
    return np.asarray(instances)  # [0]+instances[1]+instances[2]+instances[3]+instances[4]+instances[5]


def create_distributions_over_classes(labels, crop_size, stride_crop):
    classes = [[[] for i in range(0)] for i in range(NUM_CLASSES)]

    for k in xrange(len(labels)):
        # print labels[k].shape
        w, h = labels[k].shape

        for i in xrange(0, w, stride_crop):
            for j in xrange(0, h, stride_crop):
                cur_map = k
                cur_x = i
                cur_y = j
                patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if len(patch_class) != crop_size and len(patch_class[0]) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class) != crop_size:
                    cur_x = cur_x - (crop_size - len(patch_class))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]
                elif len(patch_class[0]) != crop_size:
                    cur_y = cur_y - (crop_size - len(patch_class[0]))
                    patch_class = labels[cur_map][cur_x:cur_x + crop_size, cur_y:cur_y + crop_size]

                if patch_class.shape == (crop_size, crop_size):
                    count = np.bincount(patch_class.astype(int).flatten())
                    classes[int(np.argmax(count))].append((cur_map, cur_x, cur_y))
                else:
                    print BatchColors.FAIL + "Error create_distributions_over_classes: Current patch size is " + str(
                        len(patch_class)) + "x" + str(len(patch_class[0])) + BatchColors.ENDC
                    return

    for i in xrange(len(classes)):
        print BatchColors.OKBLUE + 'Class ' + str(i + 1) + ' has length ' + str(len(classes[i])) + BatchColors.ENDC

    return classes


def create_rotation_distribution(training_class_distribution):
    rotation = [[[] for i in range(0)] for i in range(NUM_CLASSES)]

    for i in xrange(len(training_class_distribution)):
        # print len(training_class_distribution[i])
        rotation[i] = np.random.randint(0, 360, size=len(training_class_distribution[i]))

    for i in xrange(len(training_class_distribution)):
        print BatchColors.OKBLUE + 'Class ' + str(i + 1) + ' has length ' + str(
            len(training_class_distribution[i])) + ' and rotation length ' + str(len(rotation[i])) + BatchColors.ENDC
    return rotation


def create_prediction_map(img_name, prob_img, size_tuple):
    im_array = np.empty([size_tuple[0], size_tuple[1], 3], dtype=np.uint8)

    for i in xrange(size_tuple[0]):
        for j in xrange(size_tuple[1]):
            im_array[i, j, :] = retrieve_RGB_using_class(int(prob_img[i][j]))

    img = Image.fromarray(im_array)
    img.save(img_name)


def calc_accuracy_by_crop(true_crop, pred_crop, track_conf_matrix, masks=None):
    b, h, w = pred_crop.shape

    acc = 0
    local_conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    # count = 0
    for i in xrange(b):
        for j in xrange(h):
            for k in xrange(w):
                if masks is None or (masks is not None and masks[i, j, k]):
                    # count += 1
                    if true_crop[i, j, k] == pred_crop[i, j, k]:
                        acc = acc + 1
                    track_conf_matrix[true_crop[i, j, k]][pred_crop[i, j, k]] += 1
                    local_conf_matrix[true_crop[i, j, k]][pred_crop[i, j, k]] += 1

    # print count, b*h*w
    return acc, local_conf_matrix


def calc_accuracy_by_map(test_mask_data, prob_im_argmax):
    b, h, w, arg = test_mask_data.shape
    acc = 0
    conf_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

    for i in xrange(b):
        for j in xrange(h):
            for k in xrange(w):
                if test_mask_data[i][j][k] == prob_im_argmax[j][k]:
                    acc = acc + 1
                conf_matrix[test_mask_data[i][j][k][0]][prob_im_argmax[j][k]] += 1

    # print 'count', count
    return acc, conf_matrix


def select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, is_loss_or_acc='acc',
                           patch_chosen_values=None, debug=False):
    # if 0 in patch_occur:
    patch_occur[np.where(patch_occur == 0)] = 1
    patch_mean = patch_acc_loss / patch_occur
    # print is_loss_or_acc

    if is_loss_or_acc == 'acc':
        argmax_acc = np.argmax(patch_mean)
        if distribution_type == 'multi_fixed':
            cur_patch_val = int(values[argmax_acc])
        elif distribution_type == 'uniform' or distribution_type == 'multinomial':
            cur_patch_val = values[0] + argmax_acc

        if patch_chosen_values is not None:
            patch_chosen_values[int(argmax_acc)] += 1

        if debug is True:
            print 'patch_acc_loss', patch_acc_loss
            print 'patch_occur', patch_occur
            print 'patch_mean', patch_mean
            print 'argmax_acc', argmax_acc

            print 'specific', argmax_acc, patch_acc_loss[argmax_acc], patch_occur[argmax_acc], patch_mean[argmax_acc]

    elif is_loss_or_acc == 'loss':
        arg_sort_out = np.argsort(patch_mean)

        if debug is True:
            print 'patch_acc_loss', patch_acc_loss
            print 'patch_occur', patch_occur
            print 'patch_mean', patch_mean
            print 'arg_sort_out', arg_sort_out
        if distribution_type == 'multi_fixed':
            for i in xrange(len(values)):
                if patch_occur[arg_sort_out[i]] > 0:
                    cur_patch_val = int(values[arg_sort_out[i]])  # -1*(i+1)
                    if patch_chosen_values is not None:
                        patch_chosen_values[arg_sort_out[i]] += 1
                    if debug is True:
                        print 'specific', arg_sort_out[i], patch_acc_loss[arg_sort_out[i]], patch_occur[
                            arg_sort_out[i]], patch_mean[arg_sort_out[i]]
                    break
        elif distribution_type == 'uniform' or distribution_type == 'multinomial':
            for i in xrange(values[-1] - values[0] + 1):
                if patch_occur[arg_sort_out[i]] > 0:
                    cur_patch_val = values[0] + arg_sort_out[i]
                    if patch_chosen_values is not None:
                        patch_chosen_values[arg_sort_out[i]] += 1
                    if debug is True:
                        print 'specific', arg_sort_out[i], patch_acc_loss[arg_sort_out[i]], patch_occur[
                            arg_sort_out[i]], patch_mean[arg_sort_out[i]]
                    break

    if debug is True:
        print 'Current patch size ', cur_patch_val
        if patch_chosen_values is not None:
            print 'Distr of chosen sizes ', patch_chosen_values

    return cur_patch_val


#############################################################################################################


'''TensorFlow'''


#############################################################################################################


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


def dilated_icpr_original(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25
    # print x.get_shape()

    conv1 = _conv_layer(x, [5, 5, channels, 64], "main_conv1", weight_decay, is_training, rate=1)
    # pool1 = _max_pool(conv1, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool1')

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], "main_conv2", weight_decay, is_training, rate=1)
    # pool2 = _max_pool(conv2, kernel=[1, 2, 2, 1], strides=[1, 2, 2, 1], name='pool2')

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], "main_conv3", weight_decay, is_training, rate=2)
    # pool3 = _max_pool(conv3, kernel=[1, 2, 2, 1], strides=[1, 1, 1, 1], name='pool3')

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "main_conv4", weight_decay, is_training, rate=2)
    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "main_conv5", weight_decay, is_training, rate=4)
    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "main_conv6", weight_decay, is_training, rate=4)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_small(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training, rate=1)

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


def dilated_icpr_rate6_nodilation(x, dropout, is_training, weight_decay, crop_size, channels, batch_norm=True):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training,
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


def dilated_icpr_rate6(x, dropout, is_training, weight_decay, crop_size, channels, batch_norm=True):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training, rate=1, batch_norm=batch_norm)

    conv2 = _conv_layer(conv1, [5, 5, 64, 64], 'conv2', weight_decay, is_training, rate=2, batch_norm=batch_norm)

    conv3 = _conv_layer(conv2, [4, 4, 64, 128], "conv3", weight_decay, is_training, rate=3, batch_norm=batch_norm)

    conv4 = _conv_layer(conv3, [4, 4, 128, 128], "conv4", weight_decay, is_training, rate=4, batch_norm=batch_norm)

    conv5 = _conv_layer(conv4, [3, 3, 128, 256], "conv5", weight_decay, is_training, rate=5, batch_norm=batch_norm)

    conv6 = _conv_layer(conv5, [3, 3, 256, 256], "conv6", weight_decay, is_training, rate=6, batch_norm=batch_norm)

    with tf.variable_scope('conv_classifier') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[1, 1, 256, NUM_CLASSES],
                                             ini=tf.contrib.layers.xavier_initializer_conv2d(dtype=tf.float32),
                                             weight_decay=weight_decay)
        biases = _variable_on_cpu('biases', [NUM_CLASSES], tf.constant_initializer(0.0))

        conv = tf.nn.conv2d(conv6, kernel, [1, 1, 1, 1], padding='SAME')
        conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)

    return conv_classifier


def dilated_icpr_rate6_densely(x, dropout, is_training, weight_decay, crop_size, channels, batch_norm=True):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, channels, 32], "conv1", weight_decay, is_training, rate=1, batch_norm=batch_norm)

    conv2 = _conv_layer(conv1, [5, 5, 32, 32], 'conv2', weight_decay, is_training, rate=2, batch_norm=batch_norm)
    try:
        c1 = tf.concat([conv1, conv2], 3)  # c1 = 32+32 = 64
    except:
        c1 = tf.concat(concat_dim=3, values=[conv1, conv2])

    conv3 = _conv_layer(c1, [4, 4, 64, 64], "conv3", weight_decay, is_training, rate=3, batch_norm=batch_norm)
    try:
        c2 = tf.concat([c1, conv3], 3)  # c2 = 64+64 = 128
    except:
        c2 = tf.concat(concat_dim=3, values=[c1, conv3])

    conv4 = _conv_layer(c2, [4, 4, 128, 64], "conv4", weight_decay, is_training, rate=4, batch_norm=batch_norm)
    try:
        c3 = tf.concat([c2, conv4], 3)  # c3 = 128+64 = 192
    except:
        c3 = tf.concat(concat_dim=3, values=[c2, conv4])

    conv5 = _conv_layer(c3, [3, 3, 192, 128], "conv5", weight_decay, is_training, rate=5, batch_norm=batch_norm)
    try:
        c4 = tf.concat([c3, conv5], 3)  # c4 = 192+128 = 320
    except:
        c4 = tf.concat(concat_dim=3, values=[c3, conv5])

    conv6 = _conv_layer(c4, [3, 3, 320, 128], "conv6", weight_decay, is_training, rate=6, batch_norm=batch_norm)
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


def dilated_grsl(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training, rate=1, activation='lrelu')
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


def dilated_grsl_rate8(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training, rate=1, activation='lrelu')
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


def dilated_icpr_rate6_SE(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training, rate=1)
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


def dilated_icpr_rate6_squeeze(x, dropout, is_training, weight_decay, crop_size, channels):
    # Reshape input_data picture
    x = tf.reshape(x, shape=[-1, crop_size, crop_size, channels])  # default: 25x25

    conv1 = _conv_layer(x, [5, 5, channels, 64], "conv1", weight_decay, is_training, rate=1)  # 4800
    conv2 = _squeeze_conv_layer(conv1, 64, 64, 32, 5, 'conv2', weight_decay, is_training, rate=2)  # 28672

    conv3 = _squeeze_conv_layer(conv2, 64, 128, 64, 4, "conv3", weight_decay, is_training, rate=3)  # 73728
    conv4 = _squeeze_conv_layer(conv3, 128, 128, 64, 4, "conv4", weight_decay, is_training, rate=4)  # 77824

    conv5 = _squeeze_conv_layer(conv4, 128, 256, 64, 3, "conv5", weight_decay, is_training, rate=5)  # 90112
    conv6 = _squeeze_conv_layer(conv5, 256, 256, 128, 3, "conv6", weight_decay, is_training, rate=6)  # 196608

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


def test(testing_data, testing_labels, testing_instances, batch_size, weight_decay, mean_full, std_full, update_type,
         distribution_type, values, patch_acc_loss, patch_occur, net_type, former_model_path):
    print BatchColors.OKGREEN + "TESTING" + BatchColors.ENDC
    channels = testing_data.shape[-1]

    # TEST NETWORK
    crop = tf.placeholder(tf.int32)
    x = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.float32, [None, None])

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    # CONVNET
    if net_type == 'dilated_icpr_original':
        logits = dilated_icpr_original(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_grsl':
        logits = dilated_grsl(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6':
        logits = dilated_icpr_rate6(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6_small':
        logits = dilated_icpr_rate6_small(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6_SE':
        logits = dilated_icpr_rate6_SE(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6_squeeze':
        logits = dilated_icpr_rate6_squeeze(x, keep_prob, is_training, weight_decay, crop, channels)
    else:
        print BatchColors.FAIL + 'Error! Net type not identified: ' + net_type + BatchColors.ENDC
        return

    # Evaluate model
    pred_up = tf.argmax(logits, dimension=3)

    if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
        crop_size = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
                                           debug=True)
    else:
        crop_size = int(values[0])
    stride_crop = int(math.floor(crop_size / 2.0))

    # restore
    saver_restore = tf.train.Saver()

    all_cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    all_kappa = np.zeros((len(testing_data)), dtype=np.float32)
    all_f1 = np.zeros((len(testing_data)), dtype=np.float32)

    with tf.Session() as sess:
        print BatchColors.OKBLUE + 'Model restored from ' + former_model_path + '!' + BatchColors.ENDC
        saver_restore.restore(sess, former_model_path)

        for k in xrange(len(testing_data)):
            # print testing_data[k].shape
            h, w, c = testing_data[k].shape

            instaces_stride_h = (
                int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
                    ((h - crop_size) / stride_crop)) + 2)
            instaces_stride_w = (
                int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
                    ((w - crop_size) / stride_crop)) + 2)
            instaces_stride = instaces_stride_h * instaces_stride_w
            # print '--', instaces_stride, (instaces_stride/batch_size)
            # ((instaces_stride/batch_size)+1 if instaces_stride%batch_size != 0 else (instaces_stride/batch_size))

            prob_im = np.zeros([h, w, NUM_CLASSES], dtype=np.float32)
            occur_im = np.zeros([h, w, NUM_CLASSES], dtype=np.uint32)

            for i in xrange(0, (int(instaces_stride / batch_size) + 1 if instaces_stride % batch_size != 0 else int(
                        instaces_stride / batch_size))):
                test_patches, test_classes, pos = create_patches_per_map(testing_data[k], testing_labels[k], crop_size,
                                                                         stride_crop, i, batch_size)
                normalize_images(test_patches, mean_full, std_full)
                # raw_input_data("press")

                bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
                by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))

                _pred_up, _logits = sess.run([pred_up, logits], feed_dict={x: bx, y: by, crop: crop_size, keep_prob: 1.,
                                                                           is_training: False})
                for j in xrange(len(_logits)):
                    prob_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
                    int(pos[j][1]):int(pos[j][1]) + crop_size, :] += _logits[j, :, :, :]
                    occur_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
                    int(pos[j][1]):int(pos[j][1]) + crop_size, :] += 1
                    # index += 1

            occur_im[np.where(occur_im == 0)] = 1
            # np.save(output_path + 'prob_map' + str(testing_instances[k]) + '.npy', prob_im/occur_im.astype(float))
            prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=2)
            # print np.bincount(prob_im_argmax.astype(int).flatten())
            # create_prediction_map(output_path+"predImg_"+testing_instances[k]+".jpeg",
            # prob_im_argmax, testing_labels[k].shape)

            cm_test_per_map = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
            for t in xrange(h):
                for r in xrange(w):
                    # print testing_labels[k][t][r]
                    # print prob_im_argmax[t][r]
                    cm_test_per_map[int(testing_labels[k, t, r])][int(prob_im_argmax[t, r])] += 1
                    all_cm_test[int(testing_labels[k, t, r])][int(prob_im_argmax[t, r])] += 1

            _sum = 0.0
            total = 0
            for i in xrange(len(cm_test_per_map)):
                _sum += (
                    cm_test_per_map[i][i] / float(np.sum(cm_test_per_map[i])) if np.sum(cm_test_per_map[i]) != 0 else 0)
                total += cm_test_per_map[i][i]

            cur_kappa = cohen_kappa_score(testing_labels[k].flatten(), prob_im_argmax.flatten())
            cur_f1 = f1_score(testing_labels[k].flatten(), prob_im_argmax.flatten(), average='micro')
            all_kappa[k] = cur_kappa
            all_f1[k] = cur_f1

            print(" -- Test Map " + testing_instances[k] + ": Overall Accuracy= " + str(total) +
                  " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(cm_test_per_map))) +
                  " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
                  " F1 Score= " + "{:.4f}".format(cur_f1) +
                  " Kappa= " + "{:.4f}".format(cur_kappa) +
                  " Confusion Matrix= " + np.array_str(cm_test_per_map).replace("\n", "")
                  )

        _sum = 0.0
        total = 0
        for i in xrange(len(all_cm_test)):
            _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
            total += all_cm_test[i][i]

        print(" -- Test ALL MAPS: Overall Accuracy= " + str(total) +
              " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
              " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
              " F1 Score= " + np.array_str(all_f1).replace("\n", " ") +
              " Mean F1 Score= " + "{:.6f}".format(np.sum(all_f1) / float(len(testing_data))) +
              " Kappa= " + np.array_str(all_kappa).replace("\n", " ") +
              " Mean Kappa Score= " + "{:.6f}".format(np.sum(all_kappa) / float(len(testing_data))) +
              " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
              )


def validate_test(sess, testing_data, testing_labels, testing_instances, batch_size, mean_full, std_full, x, y, crop,
                  keep_prob, is_training, pred_up, logits, crop_size, step, output_path):
    stride_crop = int(math.floor(crop_size / 2.0))
    all_cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    all_kappa = np.zeros((len(testing_data)), dtype=np.float32)
    all_f1 = np.zeros((len(testing_data)), dtype=np.float32)
    all_f1_per_class = np.zeros((NUM_CLASSES), dtype=np.float32)

    for k in xrange(len(testing_data)):
        # print testing_data[k].shape
        h, w, c = testing_data[k].shape

        instaces_stride_h = (int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
            ((h - crop_size) / stride_crop)) + 2)
        instaces_stride_w = (int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
            ((w - crop_size) / stride_crop)) + 2)
        instaces_stride = instaces_stride_h * instaces_stride_w
        # print '--', instaces_stride, (instaces_stride/batch_size)
        # ((instaces_stride/batch_size)+1 if instaces_stride%batch_size != 0 else (instaces_stride/batch_size))

        prob_im = np.zeros([h, w, NUM_CLASSES], dtype=np.float32)
        occur_im = np.zeros([h, w, NUM_CLASSES], dtype=np.uint32)

        for i in xrange(0, (int(instaces_stride / batch_size) + 1 if instaces_stride % batch_size != 0 else int(
                    instaces_stride / batch_size))):
            test_patches, test_classes, pos = create_patches_per_map(testing_data[k], testing_labels[k], crop_size,
                                                                     stride_crop, i, batch_size)
            normalize_images(test_patches, mean_full, std_full)
            # raw_input("press")

            bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
            by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))

            _pred_up, _logits = sess.run([pred_up, logits],
                                         feed_dict={x: bx, y: by, crop: crop_size, keep_prob: 1., is_training: False})
            for j in xrange(len(_logits)):
                prob_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
                        int(pos[j][1]):int(pos[j][1]) + crop_size, :] += _logits[j, :, :, :]
                occur_im[int(pos[j][0]):int(pos[j][0]) + crop_size, int(pos[j][1]):int(pos[j][1]) + crop_size, :] += 1
                # index += 1

        occur_im[np.where(occur_im == 0)] = 1
        # np.save(output_path + 'prob_map' + str(testing_instances[k]) + '.npy', prob_im/occur_im.astype(float))
        prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=2)
        # print np.bincount(prob_im_argmax.astype(int).flatten())
        # create_prediction_map(output_path + "predImg_" + testing_instances[k] + "_step_" + str(step) + ".jpeg",
        # prob_im_argmax, testing_labels[k].shape)

        cm_test_per_map = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
        for t in xrange(h):
            for r in xrange(w):
                # print testing_labels[k][t][r]
                # print prob_im_argmax[t][r]
                if int(testing_labels[k][t][r]) != 6:  # not eroded
                    cm_test_per_map[int(testing_labels[k][t, r])][int(prob_im_argmax[t, r])] += 1
                    all_cm_test[int(testing_labels[k][t, r])][int(prob_im_argmax[t, r])] += 1

        _sum = 0.0
        total = 0
        for i in xrange(len(cm_test_per_map)):
            _sum += (
                cm_test_per_map[i][i] / float(np.sum(cm_test_per_map[i])) if np.sum(cm_test_per_map[i]) != 0 else 0)
            total += cm_test_per_map[i][i]

        cur_kappa = cohen_kappa_score(testing_labels[k][testing_labels[k] != 6],
                                      prob_im_argmax[testing_labels[k] != 6])
        cur_f1 = f1_score(testing_labels[k][testing_labels[k] != 6],
                          prob_im_argmax[testing_labels[k] != 6], average='macro')
        cur_f1_per_class = f1_score(testing_labels[k][testing_labels[k] != 6],
                                    prob_im_argmax[testing_labels[k] != 6], average=None)
        all_kappa[k] = cur_kappa
        all_f1[k] = cur_f1
        if len(cur_f1_per_class) == 5:  # in this case, the current image has no background class
            cur_f1_per_class = np.append(cur_f1_per_class, 0.0)

        all_f1_per_class += cur_f1_per_class

        print("---- Iter " + str(step) +
              " -- Test Map " + testing_instances[k] + ": Overall Accuracy= " + str(total) +
              " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(cm_test_per_map))) +
              " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
              " F1 Score per class= " + np.array_str(cur_f1_per_class).replace("\n", "") +
              " F1 Score= " + "{:.4f}".format(cur_f1) +
              " Kappa= " + "{:.4f}".format(cur_kappa) +
              " Confusion Matrix= " + np.array_str(cm_test_per_map).replace("\n", "")
              )

    _sum = 0.0
    total = 0
    for i in xrange(len(all_cm_test)):
        _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
        total += all_cm_test[i][i]

    print("---- Iter " + str(step) +
          " -- Test ALL MAPS: Overall Accuracy= " + str(total) +
          " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
          " F1 Score= " + np.array_str(all_f1).replace("\n", " ") +
          " Mean F1 Score= " + "{:.6f}".format(np.sum(all_f1) / float(len(testing_data))) +
          " F1 Score per class= " + np.array_str(all_f1_per_class / float(len(testing_data))).replace("\n", "") +
          " Kappa= " + np.array_str(all_kappa).replace("\n", " ") +
          " Mean Kappa Score= " + "{:.6f}".format(np.sum(all_kappa) / float(len(testing_data))) +
          " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
          )


def validate_test_multiscale(sess, testing_data, testing_labels, testing_instances, batch_size,
                             mean_full, std_full, x, y, crop, keep_prob, is_training, pred_up, logits,
                             step, distribution_type, values, update_type, num_scales, old_style, output_path):
    bkp_values = values
    all_cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    all_kappa = np.zeros((len(testing_data)), dtype=np.float32)
    all_f1 = np.zeros((len(testing_data)), dtype=np.float32)
    all_f1_per_class = np.zeros((NUM_CLASSES), dtype=np.float32)

    for k in xrange(len(testing_data)):
        if old_style is False:
            patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(step) + '.npy')
            patch_occur = np.load(output_path + 'patch_occur_step_' + str(step) + '.npy')
        else:
            patch_acc_loss = np.load(output_path + 'errorAcc_step_' + str(step) + '.npy')
            patch_occur = np.load(output_path + 'errorOccur_step_' + str(step) + '.npy')
        # patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(step) + '.npy')
        values = bkp_values
        # print testing_data[k].shape
        h, w, c = testing_data[k].shape
        mean_prob = np.zeros([num_scales, h, w, NUM_CLASSES], dtype=np.float32)

        for scale in xrange(num_scales):
            # Evaluate model
            if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or \
                            distribution_type == 'multinomial':
                crop_size = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur,
                                                   update_type, debug=True)
                patch_mean = patch_acc_loss / patch_occur
                ind = np.where(values == crop_size)
            else:
                crop_size = int(values[0])
            stride_crop = int(math.floor(crop_size / 2.0))

            instaces_stride_h = (int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
                ((h - crop_size) / stride_crop)) + 2)
            instaces_stride_w = (int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
                ((w - crop_size) / stride_crop)) + 2)
            instaces_stride = instaces_stride_h * instaces_stride_w
            # print '--', instaces_stride, (instaces_stride/batch_size)
            # ((instaces_stride/batch_size)+1 if instaces_stride%batch_size != 0 else (instaces_stride/batch_size))

            prob_im = np.zeros([h, w, NUM_CLASSES], dtype=np.float32)
            occur_im = np.zeros([h, w, NUM_CLASSES], dtype=np.uint32)

            for i in xrange(0, (int(instaces_stride / batch_size) + 1 if instaces_stride % batch_size != 0 else int(
                        instaces_stride / batch_size))):
                test_patches, test_classes, pos = create_patches_per_map(testing_data[k], testing_labels[k], crop_size,
                                                                         stride_crop, i, batch_size)
                normalize_images(test_patches, mean_full, std_full)

                bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
                by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))

                _pred_up, _logits = sess.run([pred_up, logits],
                                             feed_dict={x: bx, y: by, crop: crop_size, keep_prob: 1.,
                                                        is_training: False})
                for j in xrange(len(_logits)):
                    prob_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
                            int(pos[j][1]):int(pos[j][1]) + crop_size, :] += _logits[j, :, :, :]
                    occur_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
                             int(pos[j][1]):int(pos[j][1]) + crop_size, :] += 1
                    # index += 1

            weight = patch_mean[ind]
            occur_im[np.where(occur_im == 0)] = 1
            mean_prob[scale] = prob_im / occur_im.astype(float)
            mean_prob[scale, :, :, :] = softmax(mean_prob[scale, :, :, :])  # *weight

            values = np.delete(values, ind)
            patch_acc_loss = np.delete(patch_acc_loss, ind)
            patch_occur = np.delete(patch_occur, ind)

        prob_im_argmax = np.argmax(np.sum(mean_prob, axis=0), axis=2)
        cm_test_per_map = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
        for t in xrange(h):
            for r in xrange(w):
                # print testing_labels[k][t][r]
                # print prob_im_argmax[t][r]
                if int(testing_labels[k][t][r]) != 6:  # not eroded
                    cm_test_per_map[int(testing_labels[k][t, r])][int(prob_im_argmax[t, r])] += 1
                    all_cm_test[int(testing_labels[k][t, r])][int(prob_im_argmax[t, r])] += 1

        _sum = 0.0
        total = 0
        for i in xrange(len(cm_test_per_map)):
            _sum += (
                cm_test_per_map[i][i] / float(np.sum(cm_test_per_map[i])) if np.sum(cm_test_per_map[i]) != 0 else 0)
            total += cm_test_per_map[i][i]

        cur_kappa = cohen_kappa_score(testing_labels[k][testing_labels[k] != 6],
                                      prob_im_argmax[testing_labels[k] != 6])
        cur_f1 = f1_score(testing_labels[k][testing_labels[k] != 6],
                          prob_im_argmax[testing_labels[k] != 6], average='macro')
        cur_f1_per_class = f1_score(testing_labels[k][testing_labels[k] != 6],
                                    prob_im_argmax[testing_labels[k] != 6], average=None)
        all_kappa[k] = cur_kappa
        all_f1[k] = cur_f1
        if len(cur_f1_per_class) == 5:  # in this case, the current image has no background class
            cur_f1_per_class = np.append(cur_f1_per_class, 0.0)

        all_f1_per_class += cur_f1_per_class

        print("---- Iter " + str(step) +
              " -- Test Map " + testing_instances[k] + ": Overall Accuracy= " + str(total) +
              " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(cm_test_per_map))) +
              " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
              " F1 Score per class= " + np.array_str(cur_f1_per_class).replace("\n", "") +
              " F1 Score= " + "{:.4f}".format(cur_f1) +
              " Kappa= " + "{:.4f}".format(cur_kappa) +
              " Confusion Matrix= " + np.array_str(cm_test_per_map).replace("\n", "")
              )

    _sum = 0.0
    total = 0
    for i in xrange(len(all_cm_test)):
        _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
        total += all_cm_test[i][i]

    print("---- Iter " + str(step) +
          " -- Test ALL MAPS: Overall Accuracy= " + str(total) +
          " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
          " F1 Score= " + np.array_str(all_f1).replace("\n", " ") +
          " Mean F1 Score= " + "{:.6f}".format(np.sum(all_f1) / float(len(testing_data))) +
          " F1 Score per class= " + np.array_str(all_f1_per_class / float(len(testing_data))).replace("\n", "") +
          " Kappa= " + np.array_str(all_kappa).replace("\n", " ") +
          " Mean Kappa Score= " + "{:.6f}".format(np.sum(all_kappa) / float(len(testing_data))) +
          " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
          )


def test_or_validate_whole_images(former_model_path, testing_data, testing_labels,
                                  testing_instances, batch_size, weight_decay, mean_full, std_full, update_type,
                                  distribution_type, net_type, values, output_path,
                                  old_style=True, num_scales=1, eval_type='single_scale'):
    print BatchColors.OKBLUE + "Evaluation whole map!" + BatchColors.ENDC
    channels = testing_data[0].shape[-1]

    bkp_values = values
    for model in former_model_path:
        # PLACEHOLDERS
        crop = tf.placeholder(tf.int32)
        x = tf.placeholder(tf.float32, [None, None])
        y = tf.placeholder(tf.float32, [None, None])
        keep_prob = tf.placeholder(tf.float32)
        is_training = tf.placeholder(tf.bool, [], name='is_training')

        # CONVNET
        if net_type == 'dilated_icpr_original':
            logits = dilated_icpr_original(x, keep_prob, is_training, weight_decay, crop, channels)
        elif net_type == 'dilated_grsl':
            logits = dilated_grsl(x, keep_prob, is_training, weight_decay, crop, channels)
        elif net_type == 'dilated_icpr_rate6':
            logits = dilated_icpr_rate6(x, keep_prob, is_training, weight_decay, crop, channels, batch_norm=True)
        elif net_type == 'dilated_icpr_rate6_small':
            logits = dilated_icpr_rate6_small(x, keep_prob, is_training, weight_decay, crop, channels)
        elif net_type == 'dilated_icpr_rate6_densely':
            logits = dilated_icpr_rate6_densely(x, keep_prob, is_training, weight_decay, crop, channels,
                                                batch_norm=True)
        elif net_type == 'dilated_icpr_rate6_nodilation':
            logits = dilated_icpr_rate6_nodilation(x, keep_prob, is_training, weight_decay, crop, channels,
                                                   batch_norm=True)
        elif net_type == 'dilated8_grsl':
            logits = dilated_grsl_rate8(x, keep_prob, is_training, weight_decay, crop, channels)
        elif net_type == 'dilated_icpr_rate6_SE':
            logits = dilated_icpr_rate6_SE(x, keep_prob, is_training, weight_decay, crop, channels)
        elif net_type == 'dilated_icpr_rate6_squeeze':
            logits = dilated_icpr_rate6_squeeze(x, keep_prob, is_training, weight_decay, crop, channels)
        else:
            print BatchColors.FAIL + 'Error! Net type not identified: ' + net_type + BatchColors.ENDC
            return

        pred_up = tf.argmax(logits, dimension=3)

        # restore
        saver_restore = tf.train.Saver()

        with tf.Session() as sess:
            print BatchColors.OKBLUE + 'Model restored from ' + model + BatchColors.ENDC
            current_iter = int(model.split('-')[-1])
            saver_restore.restore(sess, model)

            if eval_type == 'single_scale':
                if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or \
                                distribution_type == 'multinomial':
                    if old_style is False:
                        patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(current_iter) + '.npy')
                        patch_occur = np.load(output_path + 'patch_occur_step_' + str(current_iter) + '.npy')
                    else:
                        patch_acc_loss = np.load(output_path + 'errorAcc_step_' + str(current_iter) + '.npy')
                        patch_occur = np.load(output_path + 'errorOccur_step_' + str(current_iter) + '.npy')
                    # patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(current_iter) + '.npy')
                    values = bkp_values

                for scale in xrange(num_scales):
                    # Evaluate model
                    if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or \
                                    distribution_type == 'multinomial':
                        crop_size = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur,
                                                           update_type, debug=True)
                    else:
                        crop_size = int(values[0])

                    validate_test(sess, testing_data, testing_labels, testing_instances, batch_size,
                                  mean_full, std_full, x, y, crop, keep_prob, is_training, pred_up, logits, crop_size,
                                  current_iter, output_path)

                    if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or \
                                    distribution_type == 'multinomial':
                        ind = np.where(values == crop_size)
                        values = np.delete(values, ind)
                        patch_acc_loss = np.delete(patch_acc_loss, ind)
                        patch_occur = np.delete(patch_occur, ind)
            else:
                # multi scale combination
                validate_test_multiscale(sess, testing_data, testing_labels, testing_instances,
                                         batch_size, mean_full, std_full, x, y, crop, keep_prob, is_training,
                                         pred_up, logits, current_iter, distribution_type, values,
                                         update_type, num_scales, old_style, output_path)

        tf.reset_default_graph()


def validation(sess, test_data, test_mask_data, selected_testing_instances, mean_full, std_full, batch_size, x, y, crop,
               keep_prob, is_training, pred_up, step, crop_size):
    linear = np.arange(len(selected_testing_instances))
    all_cm_test = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)
    # all_predcs = []
    # all_labels = []
    first = True

    for i in xrange(0,
                    ((len(linear) / batch_size) + 1 if len(linear) % batch_size != 0 else (len(linear) / batch_size))):
        test_patches, test_classes, _ = dynamically_create_patches(
            test_data, test_mask_data,
            selected_testing_instances[linear[i * batch_size:min(i * batch_size + batch_size, len(linear))]],
            crop_size, is_train=False)
        normalize_images(test_patches, mean_full, std_full)

        bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
        by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))

        _pred_up = sess.run(pred_up, feed_dict={x: bx, y: by, crop: crop_size, keep_prob: 1., is_training: False})

        # calc_accuracy_by_crop(test_classes, _pred_up, all_cm_test, test_masks)
        if first is True:
            all_predcs = _pred_up
            all_labels = test_classes
            first = False
        else:
            all_predcs = np.concatenate((all_predcs, _pred_up))
            all_labels = np.concatenate((all_labels, test_classes))

    print all_labels.shape, all_predcs.shape
    calc_accuracy_by_crop(all_labels, all_predcs, all_cm_test)

    _sum = 0.0
    total = 0
    for i in xrange(len(all_cm_test)):
        _sum += (all_cm_test[i][i] / float(np.sum(all_cm_test[i])) if np.sum(all_cm_test[i]) != 0 else 0)
        total += all_cm_test[i][i]

    cur_kappa = cohen_kappa_score(all_labels.flatten(), all_predcs.flatten())
    cur_f1 = f1_score(all_labels.flatten(), all_predcs.flatten(), average='macro')

    print("---- Iter " + str(step) +
          " -- Time " + str(datetime.datetime.now().time()) +
          " -- Validation: Overall Accuracy= " + str(total) +
          " Overall Accuracy= " + "{:.6f}".format(total / float(np.sum(all_cm_test))) +
          " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
          " F1 Score= " + "{:.4f}".format(cur_f1) +
          " Kappa= " + "{:.4f}".format(cur_kappa) +
          " Confusion Matrix= " + np.array_str(all_cm_test).replace("\n", "")
          )


def train(training_data, training_labels, training_class_distribution, training_rotation_distribution, testing_data,
          testing_labels, testing_class_distribution, testing_instances, lr_initial, batch_size, niter, weight_decay,
          mean_full, std_full, update_type, distribution_type, values, patch_acc_loss, patch_occur, patch_chosen_values,
          probs, resample_batch, output_path, display_step, net_type, dataset, former_model_path=None):
    print BatchColors.OKGREEN + "TRAINING" + BatchColors.ENDC
    print training_data[0].shape
    channels = training_data[0].shape[-1]
    print 'channels ', channels

    selected_training_instances = select_super_batch_instances(training_class_distribution,
                                                               training_rotation_distribution,
                                                               batch_size, super_batch=100)
    total_length = len(selected_training_instances)
    if os.path.isfile(os.getcwd() + '/dataset_' + dataset + '.npy'):
        selected_testing_instances = np.load(os.getcwd() + '/dataset_' + dataset + '.npy')
    else:
        selected_testing_instances = select_super_batch_instances(testing_class_distribution, batch_size=batch_size,
                                                                  super_batch=100)
        np.save(os.getcwd() + '/dataset_' + dataset + '.npy', selected_testing_instances)

    ###################
    epoch_number = 1000  # int(len(training_classes)/batch_size) # 1 epoch = images / batch
    val_inteval = 1000  # int(len(training_classes)/batch_size)
    real_test_interval = 99999999
    ###################

    # TRAIN NETWORK
    # Network Parameters
    dropout = 0.5  # Dropout, probability to keep units

    # PLACEHOLDERS
    crop = tf.placeholder(tf.int32)
    x = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.float32, [None, None])

    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    # CONVNET
    if net_type == 'dilated_icpr_original':
        logits = dilated_icpr_original(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_grsl':
        logits = dilated_grsl(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6':
        logits = dilated_icpr_rate6(x, keep_prob, is_training, weight_decay, crop, channels, batch_norm=True)
    elif net_type == 'dilated_icpr_rate6_small':
        logits = dilated_icpr_rate6_small(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6_densely':
        logits = dilated_icpr_rate6_densely(x, keep_prob, is_training, weight_decay, crop, channels, batch_norm=True)
    elif net_type == 'dilated_icpr_rate6_nodilation':
        logits = dilated_icpr_rate6_nodilation(x, keep_prob, is_training, weight_decay, crop, channels, batch_norm=True)
    elif net_type == 'dilated8_grsl':
        logits = dilated_grsl_rate8(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6_SE':
        logits = dilated_icpr_rate6_SE(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6_squeeze':
        logits = dilated_icpr_rate6_squeeze(x, keep_prob, is_training, weight_decay, crop, channels)
    else:
        print BatchColors.FAIL + 'Error! Net type not identified: ' + net_type + BatchColors.ENDC
        return

    # Define loss and optimizer
    loss = loss_def(logits, y)

    global_step = tf.Variable(0, name='main_global_step', trainable=False)
    lr = tf.train.exponential_decay(lr_initial, global_step, 50000, 0.5, staircase=True)
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=0.9).minimize(loss, global_step=global_step)

    # Define Metric Evaluate model
    pred_up = tf.argmax(logits, dimension=3)

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(max_to_keep=None)
    # restore
    saver_restore = tf.train.Saver()

    # Initializing the variables
    init = tf.initialize_all_variables()

    # Launch the graph
    shuffle = np.asarray(random.sample(xrange(total_length), total_length))
    epoch_counter = 1
    current_iter = 1

    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    # with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    with tf.Session() as sess:
        if 'model' in former_model_path:
            current_iter = int(former_model_path.split('-')[-1])
            print BatchColors.OKBLUE + 'Model restored from ' + former_model_path + BatchColors.ENDC
            patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(current_iter) + '.npy')
            patch_occur = np.load(output_path + 'patch_occur_step_' + str(current_iter) + '.npy')
            patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(current_iter) + '.npy')
            # print patch_acc_loss, patch_occur, patch_chosen_values
            saver_restore.restore(sess, former_model_path)
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
                cur_patch_size = int(values[cur_size_int])
            elif distribution_type == 'uniform':
                cur_patch_size = int(np.random.uniform(values[0], values[-1] + 1, 1))
                cur_size_int = cur_patch_size - values[0]
            elif distribution_type == 'multinomial':
                cur_size_int = np.random.multinomial(1, probs).argmax()
                cur_patch_size = values[0] + cur_size_int
            elif distribution_type == 'single_fixed':
                cur_patch_size = int(values[0])

            print cur_patch_size  # cur_size_int
            # print 'new batch of crop size == ', cur_patch_size
            shuffle, batch, it = select_batch(shuffle, batch_size, it, total_length)
            b_x, b_y, b_mask = dynamically_create_patches(training_data, training_labels,
                                                          selected_training_instances[batch], cur_patch_size,
                                                          is_train=True)
            normalize_images(b_x, mean_full, std_full)
            batch_x = np.reshape(b_x, (-1, cur_patch_size * cur_patch_size * b_x.shape[-1]))
            batch_y = np.reshape(b_y, (-1, cur_patch_size * cur_patch_size * 1))

            # Run optimization op (backprop)
            _, batch_loss, batch_pred_up = sess.run([optimizer, loss, pred_up],
                                                    feed_dict={x: batch_x, y: batch_y, crop: cur_patch_size,
                                                               keep_prob: dropout, is_training: True})

            acc, batch_cm_train = calc_accuracy_by_crop(b_y, batch_pred_up, epoch_cm_train, b_mask)
            epoch_mean += acc

            if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
                # print (batch_loss if update_type == 'loss' else (acc/float(np.sum(batch_cm_train))))
                patch_acc_loss[cur_size_int] += (
                    batch_loss * (epoch_counter / 10.0) if update_type == 'loss' else (
                        acc / float(np.sum(batch_cm_train))))
                # errorLoss[cur_size_int] += batch_loss*(epoch_counter/10.0)
                patch_occur[cur_size_int] += 1

            # DISPLAY TRAIN
            if step != 0 and step % display_step == 0:
                _sum = 0.0
                for i in xrange(len(batch_cm_train)):
                    _sum += (batch_cm_train[i][i] / float(np.sum(batch_cm_train[i]))
                             if np.sum(batch_cm_train[i]) != 0 else 0)

                print("Iter " + str(step) + " -- Time " + str(datetime.datetime.now().time()) +
                      " -- Training Minibatch: Loss= " + "{:.6f}".format(batch_loss) +
                      " Absolut Right Pred= " + str(int(acc)) +
                      " Overall Accuracy= " + "{:.4f}".format(acc / float(np.sum(batch_cm_train))) +
                      " Normalized Accuracy= " + "{:.4f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(batch_cm_train).replace("\n", "")
                      )

            # DISPLAY TRAIN EPOCH
            if step != 0 and step % epoch_number == 0:
                _sum = 0.0
                for i in xrange(len(epoch_cm_train)):
                    _sum += (
                    epoch_cm_train[i][i] / float(np.sum(epoch_cm_train[i])) if np.sum(epoch_cm_train[i]) != 0 else 0)

                print("-- Iter " + str(step) + " -- Training Epoch:" +
                      " Overall Accuracy= " + "{:.6f}".format(epoch_mean / float(np.sum(epoch_cm_train))) +
                      " Normalized Accuracy= " + "{:.6f}".format(_sum / float(NUM_CLASSES)) +
                      " Confusion Matrix= " + np.array_str(epoch_cm_train).replace("\n", "")
                      )

                epoch_mean = 0.0
                epoch_cm_train = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.uint32)

            # DISPLAY VALIDATION
            if step != 0 and step % val_inteval == 0:
                saver.save(sess, output_path + 'model', global_step=step)
                if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
                    np.save(output_path + 'patch_acc_loss_step_' + str(step) + '.npy', patch_acc_loss)
                    np.save(output_path + 'patch_occur_step_' + str(step) + '.npy', patch_occur)
                    np.save(output_path + 'patch_chosen_values_step_' + str(step) + '.npy', patch_chosen_values)

                if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
                    cur_patch_val = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur,
                                                           update_type, patch_chosen_values, debug=True)
                else:
                    cur_patch_val = int(values[0])

                validation(sess, testing_data, testing_labels, selected_testing_instances, mean_full, std_full,
                           batch_size, x, y, crop, keep_prob, is_training, pred_up, step, cur_patch_val)

            # DISPLAY REAL TEST
            if step != 0 and step % real_test_interval == 0:
                if ((channels == 4) or (channels == 5 and step >= 300000)):
                    validate_test(sess, testing_data, testing_labels, testing_instances, batch_size, mean_full,
                                  std_full,
                                  x, y, crop, keep_prob, is_training, pred_up, logits, update_type, distribution_type,
                                  values, patch_acc_loss, patch_occur, step, output_path)

            # EPOCH IS COMPLETE
            if min(it + batch_size, total_length) == total_length or total_length == it + batch_size:
                if epoch_counter % resample_batch == 0:
                    print 'epoch_counter ', epoch_counter
                    selected_training_instances = select_super_batch_instances(training_class_distribution,
                                                                               training_rotation_distribution,
                                                                               batch_size,
                                                                               super_batch=100)
                    total_length = len(selected_training_instances)
                epoch_counter += 1

        print("Optimization Finished!")

        # SAVE STATE
        saver.save(sess, output_path + 'model', global_step=step)
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            np.save(output_path + 'patch_acc_loss_step_' + str(step) + '.npy', patch_acc_loss)
            np.save(output_path + 'patch_occur_step_' + str(step) + '.npy', patch_occur)
            np.save(output_path + 'patch_chosen_values_step_' + str(step) + '.npy', patch_chosen_values)

        # Test: Final
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            cur_patch_val = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
                                                   patch_chosen_values,
                                                   debug=True)
        else:
            cur_patch_val = int(values[0])
        validation(sess, testing_data, testing_labels, selected_testing_instances, mean_full, std_full, batch_size, x,
                   y, crop, keep_prob, is_training, pred_up, step, cur_patch_val)

    tf.reset_default_graph()


def generate_final_maps(former_model_path, testing_data,
                        testing_instances, batch_size, weight_decay, mean_full, std_full, update_type,
                        distribution_type, net_type, values, dataset, output_path, old_style=True):
    # PLACEHOLDERS
    crop = tf.placeholder(tf.int32)
    x = tf.placeholder(tf.float32, [None, None])
    y = tf.placeholder(tf.float32, [None, None])
    keep_prob = tf.placeholder(tf.float32)
    is_training = tf.placeholder(tf.bool, [], name='is_training')

    # CONVNET
    channels = testing_data[0].shape[-1]
    if net_type == 'dilated_icpr_original':
        logits = dilated_icpr_original(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_grsl':
        logits = dilated_grsl(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6':
        logits = dilated_icpr_rate6(x, keep_prob, is_training, weight_decay, crop, channels, batch_norm=True)
    elif net_type == 'dilated_icpr_rate6_small':
        logits = dilated_icpr_rate6_small(x, keep_prob, is_training, weight_decay, crop, channels)
    elif net_type == 'dilated_icpr_rate6_densely':
        logits = dilated_icpr_rate6_densely(x, keep_prob, is_training, weight_decay, crop, channels,
                                            batch_norm=True)
    elif net_type == 'dilated_icpr_rate6_nodilation':
        logits = dilated_icpr_rate6_nodilation(x, keep_prob, is_training, weight_decay, crop, channels,
                                               batch_norm=True)
    elif net_type == 'dilated8_grsl':
        logits = dilated_grsl_rate8(x, keep_prob, is_training, weight_decay, crop, channels)
    else:
        print BatchColors.FAIL + 'Error! Net type not identified: ' + net_type + BatchColors.ENDC
        return

    pred_up = tf.argmax(logits, dimension=3)

    # restore
    saver_restore = tf.train.Saver()

    with tf.Session() as sess:
        current_iter = int(former_model_path.split('-')[-1])
        print BatchColors.OKBLUE + 'Model restored from ' + former_model_path + BatchColors.ENDC
        if old_style is False:
            patch_acc_loss = np.load(output_path + 'patch_acc_loss_step_' + str(current_iter) + '.npy')
            patch_occur = np.load(output_path + 'patch_occur_step_' + str(current_iter) + '.npy')
        else:
            patch_acc_loss = np.load(output_path + 'errorAcc_step_' + str(current_iter) + '.npy')
            patch_occur = np.load(output_path + 'errorOccur_step_' + str(current_iter) + '.npy')
        # patch_chosen_values = np.load(output_path + 'patch_chosen_values_step_' + str(current_iter) + '.npy')
        saver_restore.restore(sess, former_model_path)

        # Evaluate model
        if distribution_type == 'multi_fixed' or distribution_type == 'uniform' or distribution_type == 'multinomial':
            crop_size = select_best_patch_size(distribution_type, values, patch_acc_loss, patch_occur, update_type,
                                               debug=True)
        else:
            crop_size = int(values[0])
        stride_crop = int(math.floor(crop_size / 2.0))

        for k in xrange(len(testing_data)):
            # print testing_data[k].shape
            h, w, c = testing_data[k].shape

            instaces_stride_h = (
            int(((h - crop_size) / stride_crop)) + 1 if ((h - crop_size) % stride_crop) == 0 else int(
                ((h - crop_size) / stride_crop)) + 2)
            instaces_stride_w = (
            int(((w - crop_size) / stride_crop)) + 1 if ((w - crop_size) % stride_crop) == 0 else int(
                ((w - crop_size) / stride_crop)) + 2)
            instaces_stride = instaces_stride_h * instaces_stride_w
            # print '--', instaces_stride, (instaces_stride/batch_size)
            # ((instaces_stride/batch_size)+1 if instaces_stride%batch_size != 0 else (instaces_stride/batch_size))

            prob_im = np.zeros([h, w, NUM_CLASSES], dtype=np.float32)
            occur_im = np.zeros([h, w, NUM_CLASSES], dtype=np.uint32)

            for i in xrange(0, (int(instaces_stride / batch_size) + 1 if instaces_stride % batch_size != 0 else int(
                        instaces_stride / batch_size))):
                test_patches, test_classes, pos = create_patches_per_map(testing_data[k], testing_data[k][:, :, 0],
                                                                         crop_size, stride_crop, i, batch_size)
                normalize_images(test_patches, mean_full, std_full)

                bx = np.reshape(test_patches, (-1, crop_size * crop_size * test_patches.shape[-1]))
                by = np.reshape(test_classes, (-1, crop_size * crop_size * 1))

                _pred_up, _logits = sess.run([pred_up, logits],
                                             feed_dict={x: bx, y: by, crop: crop_size, keep_prob: 1.,
                                                        is_training: False})
                for j in xrange(len(_logits)):
                    prob_im[int(pos[j][0]):int(pos[j][0]) + crop_size,
                    int(pos[j][1]):int(pos[j][1]) + crop_size, :] += _logits[j, :, :, :]
                    occur_im[int(pos[j][0]):int(pos[j][0]) + crop_size, int(pos[j][1]):int(pos[j][1]) + crop_size,
                    :] += 1
                    # index += 1

            occur_im[np.where(occur_im == 0)] = 1
            # np.save(output_path + 'prob_map' + str(testing_instances[k]) + '.npy', prob_im/occur_im.astype(float))
            prob_im_argmax = np.argmax(prob_im / occur_im.astype(float), axis=2)
            if  dataset == "vaihingen":
                create_prediction_map(output_path + "top_mosaic_09cm_area" + testing_instances[k] + "_class.tif",
                                      prob_im_argmax, testing_data[k].shape)
            else:
                create_prediction_map(output_path + "top_potsdam_" + testing_instances[k] + "_label.tif",
                                      prob_im_argmax, testing_data[k].shape)

    tf.reset_default_graph()


'''
Volpi & Tuia -- Vaihingen
trainingInstances= 1,3,5,7,13,17,21,23,26,32,37
testing_instances= 11,15,28,30,34
1,3,5,7,13,17,21,23,26,32,37 11,15,28,30,34
final instances = 2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38

Potsdam Most Common:::: 2_11, 2_12, 4_10, 5_11, 6_7, 7_8, 7_10
Vaihingen Common:::: 11,15,28,30,34

python isprs_dilated_random.py /home/ISPRS/Vaihingen/ /home/segmentation_tensorflow/ISPRS/ /home/
1,3,5,7,13,17,21,23,26,32,37 11,15,28,30,34 0.01 0.001 128 150000 25 5 dilated_icpr_original single_fixed 25 loss 

python isprs_dilated_random.py /home/ISPRS/Postdam/ /home/segmentation_tensorflow/tuia/tuia_pixelwise/postdam/ 
2_10,2_11,3_10,3_11,4_10,4_11,5_10,5_11,6_7,6_8,6_9,6_10,6_11,7_7,7_8,7_9,7_10,7_11 2_12,3_12,4_12,5_12,6_12,7_12
0.001 128 1000 0.01 65 tuia 

final instances = 2_13,2_14,3_13,3_14,4_13,4_14,4_15,5_13,5_14,5_15,6_13,6_14,6_15,7_13

python isprs_dilated_random.py /home/ISPRS/Vaihingen/ 
/home/segmentation_tensorflow/ISPRS/vaihingen/all_classes/dilated8_multifixed_GRSL_acc_45_85/ 
/home/segmentation_tensorflow/ISPRS/vaihingen/all_classes/dilated8_multifixed_GRSL_acc_45_85/model-484000 
1 2,4,6,8,10,12,14,16,20,22,24,27,29,31,33,35,38 0.01 0.005 25 1000 50 25 dilated8_grsl multi_fixed 
45,55,65,75,85 acc generate_final_maps
'''


def main():
    list_params = ['input_path', 'output_path(for model, images, etc)', 'currentModelPath', 'trainingInstances',
                   'testing_instances', 'learningRate', 'weight_decay', 'batch_size', 'niter', 'reference_crop_size',
                   'reference_stride_crop',
                   'net_type[dilated_icpr_original|dilated_grsl|dilated_icpr_rate6|dilated_icpr_rate6_small|'
                   'dilated_icpr_rate6_densely]',
                   'distribution_type[single_fixed|multi_fixed|uniform|multinomial]', 'probValues',
                   'update_type [acc|loss]', 'process [training|validate_test|generate_final_maps]']
    if len(sys.argv) < len(list_params) + 1:
        sys.exit('Usage: ' + sys.argv[0] + ' ' + ' '.join(list_params))
    print_params(list_params)

    # training images path
    index = 1
    input_path = sys.argv[index]
    dataset = input_path[:-1].split("/")[-1].lower()
    # test image
    index = index + 1
    output_path = sys.argv[index]
    # current model path
    index = index + 1
    former_model_path = sys.argv[index]

    # training instances number
    index = index + 1
    trainingInstances = sys.argv[index].split(',')
    # testing instances number
    index = index + 1
    testing_instances = sys.argv[index].split(',')

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
    reference_crop_size = int(sys.argv[index])
    index = index + 1
    reference_stride_crop = int(sys.argv[index])

    index = index + 1
    net_type = sys.argv[index]

    # distr type
    index = index + 1
    distribution_type = sys.argv[index]
    index = index + 1
    values = [int(i) for i in sys.argv[index].split(',')]
    index = index + 1
    update_type = sys.argv[index]
    index = index + 1
    process = sys.argv[index]

    display_step = 50

    # wrt number of epochs -- default: 20 Vaihingen || 5 Postdam
    if dataset == 'vaihingen':
        resample_batch = 20
    elif dataset == 'postdam':
        resample_batch = 10
    else:
        print "Error! No dataset identified: ", dataset

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
    print BatchColors.WARNING + 'Reading images...' + BatchColors.ENDC
    training_data, training_labels = load_images(input_path, trainingInstances, process, image_type=dataset)
    testing_data, testing_labels = load_images(input_path, testing_instances, process, image_type=dataset)
    print training_data.shape, training_labels.shape
    print testing_data.shape, testing_labels.shape

    if process == 'training':
        print BatchColors.WARNING + 'Creating TRAINING class distribution...' + BatchColors.ENDC
        training_class_distribution = create_distributions_over_classes(training_labels, crop_size=reference_crop_size,
                                                                        stride_crop=reference_stride_crop)
        print BatchColors.WARNING + 'Creating TESTING class distribution...' + BatchColors.ENDC
        testing_class_distribution = create_distributions_over_classes(testing_labels, crop_size=reference_crop_size,
                                                                       stride_crop=reference_stride_crop)
    # print len(training_class_distribution[0])+len(training_class_distribution[1])+
    # len(training_class_distribution[2])+len(training_class_distribution[3])+len(training_class_distribution[4])+
    # len(training_class_distribution[5])
    # print len(testing_class_distribution[0])+len(testing_class_distribution[1])+
    # len(testing_class_distribution[2])+len(testing_class_distribution[3])+len(testing_class_distribution[4])+
    # len(testing_class_distribution[5])

    if os.path.isfile(os.getcwd() + '/dataset_' + dataset + '_crop_' + str(reference_crop_size) + '_stride_' + str(
            reference_stride_crop) + '_rotation.npy'):
        training_rotation_distribution = np.load(
            os.getcwd() + '/dataset_' + dataset + '_crop_' + str(reference_crop_size) + '_stride_' + str(
                reference_stride_crop) + '_rotation.npy')
        print BatchColors.OKGREEN + 'Loaded training instance rotations' + BatchColors.ENDC
    else:
        training_rotation_distribution = create_rotation_distribution(training_class_distribution)
        np.save(os.getcwd() + '/dataset_' + dataset + '_crop_' + str(reference_crop_size) + '_stride_' + str(
            reference_stride_crop) + '_rotation.npy', training_rotation_distribution)
        print BatchColors.OKGREEN + 'Created training instance rotations' + BatchColors.ENDC

    # create mean, std from training
    if os.path.isfile(os.getcwd() + '/dataset_' + dataset + '_crop_' + str(reference_crop_size) + '_stride_' + str(
            reference_stride_crop) + '_mean.npy'):
        mean_full = np.load(
            os.getcwd() + '/dataset_' + dataset + '_crop_' + str(reference_crop_size) + '_stride_' + str(
                reference_stride_crop) + '_mean.npy')
        std_full = np.load(os.getcwd() + '/dataset_' + dataset + '_crop_' + str(reference_crop_size) + '_stride_' + str(
            reference_stride_crop) + '_std.npy')
        print BatchColors.OKGREEN + 'Loaded Mean/Std from training instances' + BatchColors.ENDC
    else:
        mean_full, std_full = dynamically_calculate_mean_and_std(training_data, training_class_distribution,
                                                                 crop_size=25)
        np.save(os.getcwd() + '/dataset_' + dataset + '_crop_' + str(reference_crop_size) + '_stride_' + str(
            reference_stride_crop) + '_mean.npy', mean_full)
        np.save(os.getcwd() + '/dataset_' + dataset + '_crop_' + str(reference_crop_size) + '_stride_' + str(
            reference_stride_crop) + '_std.npy', std_full)
        print BatchColors.OKGREEN + 'Created Mean/Std from training instances' + BatchColors.ENDC

    if process == 'training':
        # print patch_acc_loss.shape, patch_occur.shape, patch_chosen_values.shape
        train(training_data, training_labels, training_class_distribution, training_rotation_distribution, testing_data,
              testing_labels, testing_class_distribution, testing_instances, lr_initial, batch_size, niter,
              weight_decay, mean_full, std_full, update_type, distribution_type, values,
              (None if distribution_type == 'single_fixed' else patch_acc_loss),
              (None if distribution_type == 'single_fixed' else patch_occur),
              (None if distribution_type == 'single_fixed' else patch_chosen_values),
              (None if distribution_type != 'multinomial' else probs), resample_batch, output_path,
              display_step, net_type, dataset, former_model_path)
    elif process == 'validate_test':
        test_or_validate_whole_images(former_model_path.split(","), testing_data, testing_labels,
                                      testing_instances, batch_size, weight_decay, mean_full, std_full, update_type,
                                      distribution_type, net_type, np.asarray(values), output_path,
                                      old_style=False, num_scales=1, eval_type='single_scale')
    elif process == 'generate_final_maps':
        generate_final_maps(former_model_path, testing_data,
                            testing_instances, batch_size, weight_decay, mean_full, std_full, update_type,
                            distribution_type, net_type, values, dataset, output_path, old_style=False)
    else:
        print BatchColors.FAIL + "Process " + process + "not found!" + BatchColors.ENDC


if __name__ == "__main__":
    main()
