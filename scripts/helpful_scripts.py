import tensorflow as tf
import typing
import os
import numpy as np
from tqdm import tqdm


# 内容特征层及loss加权系数
CONTENT_LAYERS = {"block4_conv2": 0.5, "block5_conv2": 0.5}
# 风格特征层及loss加权系数
STYLE_LAYERS = {
    "block1_conv1": 0.2,
    "block2_conv1": 0.2,
    "block3_conv1": 0.2,
    "block4_conv1": 0.2,
    "block5_conv1": 0.2,
}
# 内容图片路径
CONTENT_IMAGE_PATH = "./content/shanghai.jpg"
# 风格图片路径
STYLE_IMAGE_PATH = "./style/style_van.jpg"
# 生成图片的保存目录
OUTPUT_DIR = "./converted_content"

# 内容loss总加权系数
CONTENT_LOSS_FACTOR = 1
# 风格loss总加权系数
STYLE_LOSS_FACTOR = 100

# 图片宽度
WIDTH = 1280
# 图片高度
HEIGHT = 720

# 训练epoch数
EPOCHS = 6
# 每个epoch训练多少次
STEPS_PER_EPOCH = 100
# 学习率
LEARNING_RATE = 0.03

image_mean = tf.constant([0.485, 0.456, 0.406])
image_std = tf.constant([0.299, 0.224, 0.225])


def get_orginal_image_size(image_path):
    # load and address images
    x = tf.io.read_file(image_path)
    # decode image
    x = tf.image.decode_jpeg(x, channels=3)
    # x = imread(file_path)
    height = x.shape[0]
    width = x.shape[1]
    return height, width


def normalization(x):
    # return the normalization's value
    return (x - image_mean) / image_std


HEIGHT, WIDTH = get_orginal_image_size(CONTENT_IMAGE_PATH)


def load_images(image_path, width=WIDTH, height=HEIGHT):
    # load and address images
    x = tf.io.read_file(image_path)
    # decode image
    x = tf.image.decode_jpeg(x, channels=3)
    # resize the image size
    x = tf.image.resize(x, [height, width])
    x = x / 255
    # normalization
    x = normalization(x)
    x = tf.reshape(x, [1, height, width, 3])
    # return
    return x


# default is the filename
def split_path_and_return_file_name(file_path, index=None):
    normalized_path = os.path.normpath(file_path)
    path_commponents = normalized_path.split(os.sep)
    if index == None:
        return path_commponents[-1]
    else:
        return path_commponents[index]


def save_image(image, filename):
    x = tf.reshape(image, image.shape[1:])
    x = x * image_std + image_mean
    x = x * 255
    x = tf.cast(x, tf.int32)
    x = tf.clip_by_value(x, 0, 255)
    x = tf.cast(x, tf.uint8)
    x = tf.image.encode_jpeg(x)
    tf.io.write_file(filename, x)

