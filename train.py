
import datetime, os, glob, math, random
import tqdm
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

filters = 128
kernel_size = 3
block_size = 4
batch_size = 64
size = 64
steps_per_epoch = 100
epochs = 100
dataset = "../Datasets/Derpibooru/cropped/*.png"
example_file = "example2.png"
checkpoint_file = None

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

def gelu(x):
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def activation(x):
    return tf.nn.relu(x)

def discrete_to_continuous(x):
    return tf.cast(x, tf.float32) / 256

def continuous_to_discrete(x):
    return tf.clip_by_value(
        tf.cast(x * 256, tf.int32), 0, 255
    )

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

def scale_down(x):
    # gamma corrected Lanczos3 scaling
    return tf.maximum(tf.nn.conv2d(
        tf.nn.conv2d(
            x**2.2, 
            lanczos3_1d, [1, 2, 1, 1], 'VALID'
        ), 
        tf.transpose(lanczos3_1d, [1, 0, 2, 3]), [1, 1, 2, 1], 'VALID'
    ), 0.0) ** (1/2.2)

class Scale(tf.keras.Model):
    def __init__(self):
        super(Scale, self).__init__()
        self.convolutions = [tf.keras.layers.Conv2D(
            filters, kernel_size, 1, 'same', activation=activation
        ) for _ in range(block_size * block_size * 3)]

        self.denses = [
            tf.keras.layers.Dense(1)
            for _ in range(block_size * block_size * 3)
        ]
    
    def call(self, example):
        small = scale_down(example) * 2 - 1
        example = example[..., 5:-5, 5:-5, :] * 2 - 1
        
        example = tf.nn.space_to_depth(example, block_size)
        small = tf.nn.space_to_depth(small, block_size // 2)

        context = tf.concat([small, example], -1)

        prediction = []
        for i in range(block_size * block_size * 3):
            x = context[..., :tf.shape(small)[-1] + i]
            x = self.convolutions[i](x)
            x = self.denses[i](x)
            prediction += [x]
            
        prediction = tf.concat(prediction, -1)

        prediction = tf.nn.depth_to_space(prediction, block_size)

        return prediction * 0.5 + 0.5

    def sample(self, small):
        small = small * 2 - 1
        small = tf.nn.space_to_depth(small, block_size // 2)
        prediction = small

        for i in range(block_size * block_size * 3):
            x = prediction
            x = self.convolutions[i](x)
            x = self.denses[i](x)
            prediction = tf.concat([prediction, x], -1)
        
        prediction = prediction[..., tf.shape(small)[-1]:]

        prediction = tf.nn.depth_to_space(prediction, block_size)

        return prediction * 0.5 + 0.5

paths = glob.glob(dataset)
random.shuffle(paths)

@tf.function
def load_example(file):
    image = tf.image.decode_png(tf.io.read_file(file))[:, :, :3]
    image = tf.image.random_crop(image, [size + 10, size + 10, 3])
    image = tf.cast(image, tf.uint8)
    return image

def prepare_example(image):
    image = discrete_to_continuous(image)
    return (image, image[..., 5:-5, 5:-5, :])

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
d = d.map(load_example, num_parallel_calls=16).cache("cache74")
d = d.map(prepare_example, num_parallel_calls=16)
d = d.repeat()
d = d.batch(batch_size).prefetch(100)


name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
log_folder = os.path.join("logs", name)
os.makedirs(log_folder)
summary_writer = tf.summary.create_file_writer(log_folder)

example = prepare_example(load_example(example_file))

model = Scale()

prediction = model(example[0][None])

loss = tf.keras.losses.MeanSquaredError()(prediction, example[1][None])

def log_sample(epochs, logs):
    fake = model.sample(example[0][None])
    prediction = model(example[0][None])

    with summary_writer.as_default():
        tf.summary.image(
            'fake', fake, epochs + 1, 4
        )
        tf.summary.image(
            'prediction', prediction, epochs + 1, 4
        )
    del fake, prediction

if checkpoint_file is not None:
    model.load_weights(checkpoint_file)
    log_sample(0, [])

model.compile(
    tf.keras.optimizers.Adam(1e-4), 
    tf.keras.losses.MeanSquaredError(),
    [tf.keras.metrics.RootMeanSquaredError()]
)

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