
import datetime, os, glob, math
import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

filters = 1024
kernel_size = 13
batch_size = 1
size = 64
steps_per_epoch = 1000
epochs = 100
dataset = "../Datasets/Derpibooru/cropped/*.png"
example_file = "example.png"

radius = kernel_size // 2

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

def join_inner_axes(tensor):
    shape = tf.shape(tensor)
    return tf.reshape(
        tensor, tf.concat([shape[:-2], [shape[-2] * shape[-1]]], 0)
    )


class AutoregressiveConvolution2D(tf.keras.layers.Layer):
    # [batch, height, width, features]
    def __init__(self, filters, kernel_size, use_bias=True):
        super(AutoregressiveConvolution2D, self).__init__()

        self.kernel_size = kernel_size
        self.radius = kernel_size // 2

        # Layers can contain other layers
        self.top = tf.keras.layers.Conv2D(
            filters, (self.radius, kernel_size), 1, 'valid', use_bias=False
        )
        self.left = tf.keras.layers.Conv2D(
            filters, (1, self.radius), 1, 'valid', use_bias=use_bias
        )

    def call(self, inputs):
        return self.top(
            tf.pad(inputs[..., :-1, :, :], [
                [0, 0], [self.radius, 0], [self.radius, self.radius], [0, 0]
            ])
        ) + self.left(
            tf.pad(inputs[..., :-1, :], [
                [0, 0], [0, 0], [self.radius, 0], [0, 0]
            ])
        )

class AutoregressiveDense(tf.keras.layers.Layer):
    # [..., depth, features]
    def __init__(self, depth, features, use_bias=True):
        super(AutoregressiveDense, self).__init__()
        self.features = features
        self.use_bias = use_bias

        self.layers = [
            tf.keras.layers.Dense(self.features, use_bias=self.use_bias) 
            for _ in range(depth)
        ]

    def call(self, inputs):
        return tf.stack([
            layer(join_inner_axes(inputs[..., :i, :]))
            for i, layer in enumerate(self.layers)
        ], -2)
        # [..., depth, features]

class Scale(tf.keras.Model):
    def __init__(self):
        super(Scale, self).__init__()
        self.convolution = AutoregressiveConvolution2D(
            filters, kernel_size, False
        )
        self.dense = AutoregressiveDense(3, filters, True)
        self.final = [tf.keras.layers.Dense(256) for _ in range(3)]
    
    def call(self, inputs):
        # inputs is int32 [batch, height, width, channels]
        x = tf.one_hot(inputs, 256)

        # TODO: gather might be more efficient
        flattened = join_inner_axes(x)
        spatial = self.convolution(flattened)[..., None, :]
        channels = self.dense(x)
        # [batch, height, width, channels, features]

        x = spatial + channels
        
        x = tf.nn.relu(x)
        x = tf.stack(
            [f(x) for f, x in zip(self.final, tf.unstack(x, axis=-2))], -2
        )
        return x

    def sample(self, size):
        # size = [..., batches, height, width]
        # [..., height, width, channels * values]
        @tf.function
        def pixel(variables, top_features):
            left_values, sample = variables

            top_left_features = \
                top_features[..., None, :] + self.convolution.left(left_values)

            channels = tf.zeros(size[:-2] + [1, 1, 0])
            samples = []

            for c in range(3):
                features = top_left_features + self.dense.layers[c](
                    channels
                )
                features = tf.nn.relu(features)
                logits = self.final[c](features)
                
                sample = tfp.distributions.Categorical(logits).sample()
                value = tf.one_hot(sample, 256)

                samples += [sample]
                channels = tf.concat([channels, value], -1)
            
            sample = tf.stack(samples, -1)
            return (tf.concat([left_values[..., 1:, :], channels], -2), sample)
        
        @tf.function
        def line(variables, context):
            top_values, sample = variables

            top_features = self.convolution.top(
                tf.pad(top_values, [[0, 0], [0, 0], [radius, radius], [0, 0]])
            )

            pixels, sample = tf.scan(
                pixel, 
                tf.transpose(top_features, [2, 0, 1, 3]), 
                (
                    tf.zeros(size[:-2] + [1, radius, 3 * 256]), 
                    tf.zeros(size[:-2] + [1, 1, 3], dtype=tf.int32)
                )
            )

            pixels = tf.transpose(pixels[..., -1, :], [1, 2, 0, 3])
            sample = tf.transpose(sample[..., -1, :], [1, 2, 0, 3])

            return (tf.concat([top_values[..., 1:, :, :], pixels], -3), sample)

        sample = tf.transpose(tf.scan(
            line,
            tf.zeros((size[-2],)),
            (
                tf.zeros(size[:-2] + [radius, size[-1], 3 * 256]), 
                tf.zeros(size[:-2] + [1, size[-1], 3], dtype=tf.int32)
            )
        )[1][..., -1, :, :], [1, 0, 2, 3])

        return sample

paths = glob.glob(dataset)

@tf.function
def load_example(file):
    image = tf.image.decode_png(tf.io.read_file(file))[:, :, :3]
    image = tf.image.random_crop(image, [size + 10, size + 10, 3])
    image = tf.cast(image, tf.int32)
    return (image, image)

lanczos3 = [
    3 * math.sin(math.pi * x) * math.sin(math.pi * x / 3) / math.pi**2 / x**2
    for x in np.linspace(-2.75, 2.75, 12)
]
lanczos3 = [x / sum(lanczos3) for x in lanczos3]
lanczos3_1d = tf.constant(
    [
        [[
            [a if o == i else 0 for o in range(3)]
            for i in range(3)
        ]] 
        for a in lanczos3
    ], dtype=tf.float32
)

d = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
d = d.map(load_example, num_parallel_calls = 16).cache()
d = d.shuffle(10000).repeat()
d = d.batch(batch_size).prefetch(100)


def load_file(file):
    image = tf.image.decode_png(tf.io.read_file(file))[:size, :size, :3]
    return tf.cast(image, tf.int32)

name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
log_folder = os.path.join("logs", name)
os.makedirs(log_folder)
summary_writer = tf.summary.create_file_writer(log_folder)

example = load_file(example_file)[None]

model = Scale()

model.compile(
    tf.keras.optimizers.Adam(), 
    tf.keras.losses.SparseCategoricalCrossentropy(True),
    [tf.keras.metrics.SparseCategoricalAccuracy()]
)

prediction = model(example)

def log_sample(epochs, logs):
    fake = model.sample([4, size, size])
    fake = tf.cast(fake, tf.float32) / 255.0

    with summary_writer.as_default():
        tf.summary.image(
            'fake', fake, epochs, 4
        )
    del fake

model.fit(
    d, steps_per_epoch=steps_per_epoch, epochs=epochs,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(log_folder, "model.{epoch:02d}.hdf5")
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=log_folder, write_graph=True, write_images=False, 
            profile_batch=0
        ),
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=log_sample
        )
    ]
)