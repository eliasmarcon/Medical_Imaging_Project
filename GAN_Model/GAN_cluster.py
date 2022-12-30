# import necessary packages

from utils_gan import *

# TODO: add commandline arguments (epoch, batch size, img output dir, model_name)
BENIGN_PATH = '../dat/benign/'
MALIGNANT_PATH = '../dat/malignant/'
save_dir_benign = "./GAN_Images/benign/"
save_dir_malignant = "./GAN_Images/malignant/"
save_dir = save_dir_benign
CHECKPOINT_PATH = './GAN_Images/training_checkpoints/'

BATCH_SIZE = 16
EPOCHS = 2
noise_dim = 100
num_examples_to_generate = 16


def main():
    files_benign = get_all_images(BENIGN_PATH)
    print("Sample size of benign images:", len(files_benign))
    files_malignant = get_all_images(MALIGNANT_PATH)
    print("Sample size of malignant images:", len(files_malignant))

    # TODO: add a switch for benign/malignant/all files being generated

    dataset = prepare_dataset(files_benign)

    generator = Generator()
    discriminator = Discriminator()
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint = create_checkpoint(CHECKPOINT_PATH, generator_optimizer, discriminator_optimizer, generator,
                                   discriminator)
    checkpoint_prefix = os.path.join(CHECKPOINT_PATH, "ckpt")

    random_vector_for_generation = tf.random.normal([num_examples_to_generate, noise_dim])

    # Training
    for epoch in range(EPOCHS):
        start = time.time()

        for images in dataset:
            noise = tf.random.normal([BATCH_SIZE, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                generated_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(generated_output)
                disc_loss = discriminator_loss(real_output, generated_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

        display.clear_output(wait=True)

        generate_and_save_images(generator, epoch + 1, random_vector_for_generation, save_dir)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:  # das auf 15 ändern

            # checkpoint.save(file_prefix = checkpoint_prefix)
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1, time.time() - start))

    # generating after the final epoch
    # display.clear_output(wait = True)
    generate_and_save_images(generator, epoch, random_vector_for_generation, save_dir)


if __name__ == '__main__':
    main()
