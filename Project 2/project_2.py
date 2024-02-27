# This is a python script for Neural style transfer

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from PIL import Image
import PIL.Image

tf.keras.backend.set_floatx('float64')

vgg_old = tf.keras.applications.VGG19(include_top=False, weights='imagenet')


class StyleContentModel(tf.keras.models.Model):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs * 255)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = outputs[:self.num_style_layers], outputs[self.num_style_layers:]
        style_outputs = [gram_matrix(layer) for layer in style_outputs]
        content_dict = {content_name: value for content_name, value in zip(self.content_layers, content_outputs)}
        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}
        return {'content': content_dict, 'style': style_dict}


def load_img(path_to_img, max_dim):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)  # Pixels are b/w 0 and 255
    img = tf.image.convert_image_dtype(img, tf.float64)  # Converts all the pixels b/w 0 and 1
    shape = tf.cast(tf.shape(img)[:-1], tf.float64)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = tf.expand_dims(img, 0)
    img = tf.cast(img, tf.float64)

    return img

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)

def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2] * input_shape[3], tf.float64)
    return result / (num_locations)


def vgg_layers(layer_names):
    layer_outputs = []
    for name in layer_names:
        layer_outputs.append(vgg_old.get_layer(name).output)
    model = tf.keras.Model(inputs=vgg_old.input, outputs=layer_outputs)
    return model


def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight):
    style_outputs = outputs['style']
    content_outputs = outputs['content']
    style_loss = tf.add_n(
        [tf.reduce_sum((style_outputs[name] - style_targets[name]) ** 2) for name in style_outputs.keys()])
    style_loss *= style_weight / (4 * len(style_outputs))
    content_loss = tf.add_n(
        [tf.reduce_sum((content_outputs[name] - content_targets[name]) ** 2) for name in content_outputs.keys()])
    content_loss *= content_weight / 2
    loss = style_loss + content_loss
    return loss

@tf.function()
def train_step(image, opt,extractor, style_targets, content_targets, style_weight, content_weight, tv_weight):
    with tf.GradientTape() as tape:
        outputs = extractor(image)
        loss = style_content_loss(outputs, style_targets, content_targets, style_weight, content_weight) + tv_weight * tf.cast(tf.image.total_variation(image)[0], tf.float64)
    grad = tape.gradient(loss, image)
    opt.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss


def main():
    # Use a breakpoint in the code line below to debug your script.
    content_image = load_img("Neural3.jpg", 512)

    style_image = load_img("vangogh.jpg", 512)
    
    content_layers = ["block4_conv2"]
    style_layers = ["block1_conv1",
                    "block2_conv1",
                    "block3_conv1",
                    "block4_conv1",
                    "block5_conv1"]

    extractor = StyleContentModel(style_layers, content_layers)
    style_targets = extractor(style_image)['style']
    content_targets = extractor(content_image)['content']

    image = tf.Variable(content_image)
    content_weight = 1e3
    style_weight = 1
    tv_weight = 1e3

    opt = tf.optimizers.Adam(learning_rate=0.02, beta_1=0.99, epsilon=1e-1)

    loops = 100
    iters_per_loop = 10

    file_writer = tf.summary.create_file_writer('logs' + f'/stw{style_weight}_cow{content_weight}')
    file_writer.set_as_default()

    for loop in range(loops):
        tf.summary.image('image', data=image, step=loop * iters_per_loop)
        for it in range(iters_per_loop):
            # YOUR CODE
            loss = train_step(image, opt, extractor, style_targets, content_targets, style_weight, content_weight)
            tf.summary.scalar('loss', data=loss, step=loop * iters_per_loop + it)
    # Save the stylised image
    img_tensor = tensor_to_image(image)
    tf.keras.utils.save_img('stylised_image.jpg', img_tensor)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
