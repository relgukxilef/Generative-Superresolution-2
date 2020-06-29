
import datetime, os, glob, math
import tqdm
import tensorflow as tf
import numpy as np

filters = 128
kernel_size = 11
batch_size = 64
size = 64

weight_decay = 0.0001

generator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same', 
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation=tf.keras.activations.softplus
    ),
    tf.keras.layers.Conv2DTranspose(
        3, 2, 2, 
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay)
    ),
])

discriminator = tf.keras.Sequential([
    tf.keras.layers.Conv2D(
        filters, kernel_size, 1, 'same',
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation=tf.keras.activations.softplus
    ),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(
        filters,
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay),
        activation=tf.keras.activations.softplus
    ),
    tf.keras.layers.Dense(
        1, 
        kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        bias_regularizer=tf.keras.regularizers.l2(weight_decay)
    ),
])

#optimizer = tf.keras.optimizers.SGD(0.001)
optimizer = tf.keras.optimizers.Adam()

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
d = d.shuffle(10000).repeat()
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

    with tf.GradientTape(persistent=True) as tape:
        fake = generator(tf.concat([
            small * 2 - 1, 
            noise
        ], axis=-1))

        real_logits = discriminator(
            tf.concat(
                [
                    tf.nn.space_to_depth(real[:, 5:-5, 5:-5, :], 2) * 2 - 1, 
                    small * 2 - 1
                ], -1
            )
        )
        fake_logits = discriminator(
            tf.concat(
                [
                    tf.nn.space_to_depth(fake, 2), 
                    small * 2 - 1
                ], -1
            )
        )
        noise_logits = discriminator(
            tf.concat(
                [
                    tf.random.uniform(
                        (batch_size, size // 2, size // 2, 12), -1.0, 1.0
                    ),
                    small * 2 - 1
                ], -1
            )
        )

        generator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
            tf.ones_like(fake_logits), fake_logits
        )
        
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

    return generator_loss, discriminator_loss

def load_file(file):
    image = tf.image.decode_png(tf.io.read_file(file))[:, :, :3]
    return tf.cast(image, tf.float32) / 255

name = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")
summary_writer = tf.summary.create_file_writer(os.path.join("logs", name))

example = load_file("example.png")
example = tf.concat([
    example * 2 - 1,
    tf.random.normal((example.shape[0], example.shape[1], 12))
], -1)

for step, image in enumerate(tqdm.tqdm(d)):
    generator_loss, discriminator_loss = train_step(image)
    
    if step % 100 == 0:
        with summary_writer.as_default():
            tf.summary.scalar('generator_loss', generator_loss, step)
            tf.summary.scalar('discriminator_loss', discriminator_loss, step)

            tf.summary.image(
                'fake', generator(example[None]) * 0.5 + 0.5, step, 1
            )