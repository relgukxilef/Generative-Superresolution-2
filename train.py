
import datetime, os, glob, math
import tqdm
import tensorflow as tf
import numpy as np

filters = 1024
kernel_size = 7
batch_size = 1
size = 64

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

class AutoregressiveConvolution2d(tf.keras.layers.Layer):
    # [batch, height, width, features]
    def __init__(self, filters, kernel_size, use_bias=True):
        super(AutoregressiveConvolution2d, self).__init__()

        self.kernel_size = kernel_size
        self.radius = kernel_size // 2

        # Layers can contain other layers
        self.top = tf.keras.layers.Conv2D(
            filters, (kernel_size, self.radius), 1, 'valid', use_bias=False
        )
        self.left = tf.keras.layers.Conv2D(
            filters, (self.radius, 1), 1, 'valid', use_bias=use_bias
        )

    def call(self, inputs):
        return self.top(
            tf.pad(inputs[..., :-1, :], [
                [0, 0], [self.radius, self.radius], [self.radius, 0], [0, 0]
            ])
        ) + self.left(
            tf.pad(inputs[..., :-1, :, :], [
                [0, 0], [self.radius, 0], [0, 0], [0, 0]
            ])
        )

class AutoregressiveDense(tf.keras.layers.Layer):
    # [..., features]
    def __init__(self, outputs, stride, use_bias=True):
        super(AutoregressiveDense, self).__init__()
        self.outputs = outputs
        self.stride = stride
        self.use_bias = use_bias

    def build(self, input_shape):
        self.layers = [
            tf.keras.layers.Dense(self.outputs, use_bias=self.use_bias) 
            for _ in range(input_shape[-1] // self.stride)
        ]

    def call(self, inputs):
        return tf.stack([
            layer(inputs[..., :i * self.stride]) 
            for i, layer in enumerate(self.layers)
        ], -2)
        # [..., depth, features]

class Scale(tf.keras.Model):
    def __init__(self):
        super(Scale, self).__init__()
        self.convolution = AutoregressiveConvolution2d(
            filters * 3, kernel_size, False
        )
        self.dense = AutoregressiveDense(filters, 256, True)
        self.final = [tf.keras.layers.Dense(256) for _ in range(3)]
    
    def call(self, inputs):
        # inputs is int32 [batch, height, width, channels]
        x = tf.one_hot(inputs, 256)

        # TODO: gather would be more efficient
        flattened = tf.reshape(x, tf.concat([tf.shape(x)[:-2], [256 * 3]], 0))
        spatial = tf.reshape(
            self.convolution(flattened), 
            tf.concat([tf.shape(x)[:-2], [3, filters]], 0)
        )
        channels = self.dense(flattened)
        # [batch, height, width, channels, values]

        x = spatial + channels
        
        x = tf.nn.relu(x)
        x = tf.stack(
            [f(x) for f, x in zip(self.final, tf.unstack(x, axis=-2))], -2
        )
        return x

paths = glob.glob("../Datasets/Derpibooru/cropped/*.png")

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
d = d.shuffle(100).repeat()
d = d.batch(batch_size).prefetch(100)


def load_file(file):
    image = tf.image.decode_png(tf.io.read_file(file))[:size, :size, :3]
    return tf.cast(image, tf.int32)

name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

example = load_file("example.png")[None]

model = Scale()
model.compile(
    tf.keras.optimizers.Adam(), 
    tf.keras.losses.SparseCategoricalCrossentropy(True),
    [tf.keras.metrics.SparseCategoricalAccuracy()]
)

prediction = model(example)

model.fit(d, steps_per_epoch = 1000)