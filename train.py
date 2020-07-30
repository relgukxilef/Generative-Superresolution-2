
import datetime, os, glob, math
import tqdm
import tensorflow as tf
import numpy as np

filters = 32
kernel_size = 3
batch_size = 64
size = 64

weight_decay = 0.0001

gpu = tf.config.experimental.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(gpu, True)

mean = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(3, use_bias=False),
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(3, use_bias=False),
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(3)
])

# factorization of convolutions reduces memory usage but slows down training
# it increases both the time per step and the number of steps

generator = tf.keras.Sequential([
    tf.keras.layers.Conv2DTranspose(
        filters, kernel_size * 2, 2, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(
        3, kernel_size, 1, 'same',
        activation=tf.keras.activations.tanh
    ),
])

discriminator = tf.keras.Sequential([
    #tf.keras.layers.Conv2DTranspose(
    #    filters, kernel_size * 2, 2, 'same'
    #),
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same'
    ),
    tf.keras.layers.LeakyReLU(),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(
        1
    ),
])

#optimizer = tf.keras.optimizers.SGD(0.001)
optimizer = tf.keras.optimizers.Adam(0.0002, 0.0)

paths = glob.glob("../Datasets/Derpibooru/cropped/*.png")

@tf.function
def load_example(file):
    image = tf.image.decode_png(tf.io.read_file(file))[:, :, :3]
    image = tf.image.random_crop(image, [size + 10, size + 10, 3])
    return tf.cast(image, tf.float32) / 255

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
d = d.shuffle(1000).repeat()
d = d.batch(batch_size).prefetch(100)

@tf.function
def train_step(real):
    # scale gamma corrected
    small = tf.maximum(tf.nn.conv2d(
        tf.nn.conv2d(
            real**2.2, 
            lanczos3_1d, [1, 2, 1, 1], 'VALID'
        ), 
        tf.transpose(lanczos3_1d, [1, 0, 2, 3]), [1, 1, 2, 1], 'VALID'
    ), 0.0) ** (1/2.2)

    noise = tf.random.normal((batch_size, size // 2, size // 2, 12))

    real = real[:, 5:-5, 5:-5, :] * 2 - 1
    small = small * 2 - 1

    with tf.GradientTape(persistent=True) as tape:
        #fake = generator(tf.concat([small, noise], axis=-1))
        fake = generator(small)

        real_logits = discriminator(tf.concat([
            tf.nn.space_to_depth(real, 2), 
            small,
            #real
        ], -1))
        fake_logits = discriminator(tf.concat([
            tf.nn.space_to_depth(fake, 2), 
            small,
            #fake
        ], -1))

        median_loss = tf.keras.losses.MeanAbsoluteError()(fake, real)

        distribution_loss = tf.keras.losses.BinaryCrossentropy(
            from_logits=True
        )(
            tf.ones_like(fake_logits), fake_logits
        )

        generator_loss = sum([
            distribution_loss,
            #median_loss,
        ])
        
        discriminator_loss = sum([
            tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.ones_like(real_logits), real_logits
            ),
            tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                tf.zeros_like(fake_logits), fake_logits
            )
        ])

    optimizer.apply_gradients(zip(
        tape.gradient(generator_loss, generator.trainable_variables), 
        generator.trainable_variables
    ))
    optimizer.apply_gradients(zip(
        tape.gradient(discriminator_loss, discriminator.trainable_variables), 
        discriminator.trainable_variables
    ))

    fake_logits_deviation = tf.math.reduce_std(fake_logits)
    real_logits_deviation = tf.math.reduce_std(real_logits)


    return (
        distribution_loss, fake_logits_deviation, 
        real_logits_deviation, discriminator_loss
    )

def load_file(file):
    image = tf.image.decode_png(tf.io.read_file(file))[:, :, :3]
    return tf.cast(image, tf.float32) / 255

name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

example = load_file("example.png")
example = tf.concat([
    example * 2 - 1,
    #tf.random.normal((example.shape[0], example.shape[1], 12))
], -1)

for step, image in enumerate(tqdm.tqdm(d)):
    distribution_loss, fake_logits_deviation, \
        real_logits_deviation, discriminator_loss = \
        train_step(image)
    
    if step % 100 == 0:
        with summary_writer.as_default():
            tf.summary.scalar('distribution_loss', distribution_loss, step)
            tf.summary.scalar(
                'fake_logits_deviation', fake_logits_deviation, step
            )
            tf.summary.scalar(
                'real_logits_deviation', real_logits_deviation, step
            )
            tf.summary.scalar('discriminator_loss', discriminator_loss, step)

            fake = generator(example[None], training=False)
            tf.summary.image(
                'fake', fake * 0.5 + 0.5, step, 1
            )