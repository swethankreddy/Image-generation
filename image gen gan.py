import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import layers
import time
import matplotlib.pyplot as plt
import os

# Constants
BATCH_SIZE = 32
NOISE_DIMENSION = 100
EPOCHS = 2000
NUM_EXAMPLES_TO_GENERATE = 16
CHECKPOINT_DIRECTORY = './training_checkpoints'

# Data generator for image preprocessing
data_generator = ImageDataGenerator(rescale=1.0 / 255.0)

# Load dataset from the specified directory
image_dataset = data_generator.flow_from_directory(
    '/content/drive/MyDrive/horse-or-human',
    target_size=(64, 64),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=True
)

def create_generator():
    """Build the generator model."""
    model = tf.keras.Sequential([
        layers.Dense(8 * 8 * 256, use_bias=False, input_shape=(NOISE_DIMENSION,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

generator_model = create_generator()

def create_discriminator():
    """Build the discriminator model."""
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[64, 64, 3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

discriminator_model = create_discriminator()

# Define loss functions
binary_cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def compute_discriminator_loss(real_output, fake_output):
    real_loss = binary_cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def compute_generator_loss(fake_output):
    return binary_cross_entropy(tf.ones_like(fake_output), fake_output)

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def execute_training_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIMENSION])
    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        generated_images = generator_model(noise, training=True)
        real_output = discriminator_model(images, training=True)
        fake_output = discriminator_model(generated_images, training=True)
        
        gen_loss = compute_generator_loss(fake_output)
        disc_loss = compute_discriminator_loss(real_output, fake_output)

    # Calculate gradients
    gradients_of_generator = generator_tape.gradient(gen_loss, generator_model.trainable_variables)
    gradients_of_discriminator = discriminator_tape.gradient(disc_loss, discriminator_model.trainable_variables)

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator_model.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator_model.trainable_variables))

def generate_and_save_images(model, epoch, test_input):
    predictions = model(test_input, training=False)
    plt.figure(figsize=(4, 4))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((predictions[i] * 0.5 + 0.5).numpy())
        plt.axis('off')
    plt.savefig(f'image_at_epoch_{epoch:04d}.png')
    plt.close()

# Generate images for the specified number of examples
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIMENSION])

# Prepare training dataset
all_images = []
for _ in range(len(image_dataset)):
    batch = next(image_dataset)
    all_images.append(batch)

all_images = np.concatenate(all_images, axis=0).astype('float32')
all_images = (all_images - 0.5) / 0.5
train_data = tf.data.Dataset.from_tensor_slices(all_images).shuffle(60000).batch(BATCH_SIZE)

# Checkpoint setup
checkpoint_prefix = os.path.join(CHECKPOINT_DIRECTORY, "ckpt")
checkpoint = tf.train.Checkpoint(generator=generator_model,
                                 discriminator=discriminator_model,
                                 generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer)

def train_model(dataset, epochs, initial_epoch=0):
    for epoch in range(initial_epoch, epochs):
        start_time = time.time()
        for image_batch in dataset:
            execute_training_step(image_batch)

        generate_and_save_images(generator_model, epoch + 1, seed)
        print(f'Time for epoch {epoch + 1} is {time.time() - start_time} sec')
        checkpoint.save(file_prefix=checkpoint_prefix)

    generate_and_save_images(generator_model, epochs, seed)

# Create checkpoint directory if it doesn't exist
if not os.path.exists(CHECKPOINT_DIRECTORY):
    os.makedirs(CHECKPOINT_DIRECTORY)
    starting_epoch = 0
else:
    starting_epoch = 0

# Start training the model
train_model(train_data, EPOCHS, starting_epoch)
