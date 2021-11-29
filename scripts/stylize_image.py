import tensorflow as tf
import os
import numpy as np
from tqdm import tqdm
from helpful_scripts import (
    CONTENT_LOSS_FACTOR,
    load_images,
    CONTENT_IMAGE_PATH,
    STYLE_IMAGE_PATH,
    split_path_and_return_file_name,
    WIDTH,
    HEIGHT,
    LEARNING_RATE,
    STYLE_LOSS_FACTOR,
    OUTPUT_DIR,
    STEPS_PER_EPOCH,
    save_image,
    EPOCHS,
)
from set_model import NeuralStyleTransferModel


#  build model
model = NeuralStyleTransferModel()
# load the content images
content_image = load_images(CONTENT_IMAGE_PATH)
# print(content_image)
# style image
style_image = load_images(STYLE_IMAGE_PATH)
# content image name
contentt_image_name = split_path_and_return_file_name(CONTENT_IMAGE_PATH)
# calculating the content feature of target content images
target_content_features = model([content_image,])["content"]
# print(target_content_features)
# calculating the style feature of style images
target_style_features = model([style_image,])["style"]

# 使用Adma优化器
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

# 基于内容图片随机生成一张噪声图片
noise_image = tf.Variable(
    (content_image + np.random.uniform(-0.2, 0.2, (1, HEIGHT, WIDTH, 3))) / 2
)


# 创建保存生成图片的文件夹
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


M = WIDTH * HEIGHT
N = 3


def _compute_content_loss(noise_features, target_features):
    """
  calculating the given layers' content loss
  noise features
  target feature
  """

    content_loss = tf.reduce_sum(tf.square(noise_features - target_features))
    # calculating coefficient
    x = 2.0 * M * N
    content_loss = content_loss / x
    return content_loss


def compute_content_loss(noise_content_features):
    """
  calculating current image's loss
  """
    # init content loss
    content_losses = []
    # calculating the weighted content loss
    for (noise_feature, factor), (target_feature, _) in zip(
        noise_content_features, target_content_features
    ):
        layer_content_loss = _compute_content_loss(noise_feature, target_feature)
        content_losses.append(layer_content_loss * factor)
    return tf.reduce_sum(content_losses)


def gram_matrix(feature):
    """
  calculating the Gram matrix
  """
    # exhange the dimension, put the dimension on the first
    x = tf.transpose(feature, perm=[2, 0, 1])
    # reshape to 2D
    x = tf.reshape(x, (x.shape[0], -1))
    return x @ tf.transpose(x)


def _compute_style_loss(noise_feature, target_feature):
    """
    计算指定层上两个特征之间的风格loss
    :param noise_feature: 噪声图片在指定层的特征
    :param target_feature: 风格图片在指定层的特征
    """
    noise_gram_matrix = gram_matrix(noise_feature)
    style_gram_matrix = gram_matrix(target_feature)
    style_loss = tf.reduce_sum(tf.square(noise_gram_matrix - style_gram_matrix))
    # 计算系数
    x = 4.0 * (M ** 2) * (N ** 2)
    return style_loss / x


def compute_style_loss(noise_style_features):
    """
    计算并返回图片的风格loss
    :param noise_style_features: 噪声图片的风格特征
    """
    style_losses = []
    for (noise_feature, factor), (target_feature, _) in zip(
        noise_style_features, target_style_features
    ):
        layer_style_loss = _compute_style_loss(noise_feature, target_feature)
        style_losses.append(layer_style_loss * factor)
    return tf.reduce_sum(style_losses)


def total_loss(noise_features):
    """
    计算总损失
    :param noise_features: 噪声图片特征数据
    """
    content_loss = compute_content_loss(noise_features["content"])
    style_loss = compute_style_loss(noise_features["style"])
    return content_loss * CONTENT_LOSS_FACTOR + style_loss * STYLE_LOSS_FACTOR


# 使用tf.function加速训练
@tf.function
def train_one_step():
    """
    一次迭代过程
    """
    # 求loss
    with tf.GradientTape() as tape:
        noise_outputs = model(noise_image)
        loss = total_loss(noise_outputs)
    # 求梯度
    grad = tape.gradient(loss, noise_image)
    # 梯度下降，更新噪声图片
    optimizer.apply_gradients([(grad, noise_image)])
    return loss


# 共训练settings.EPOCHS个epochs
for epoch in range(EPOCHS):
    # 使用tqdm提示训练进度
    with tqdm(
        total=STEPS_PER_EPOCH, desc="Epoch {}/{}".format(epoch + 1, EPOCHS)
    ) as pbar:
        # 每个epoch训练settings.STEPS_PER_EPOCH次
        for step in range(STEPS_PER_EPOCH):
            _loss = train_one_step()
            pbar.set_postfix({"loss": "%.4f" % float(_loss)})
            pbar.update(1)
        # 每个epoch保存一次图片
        save_image(
            noise_image,
            "{}/converted_{}_{}".format(OUTPUT_DIR, epoch + 1, contentt_image_name),
        )
