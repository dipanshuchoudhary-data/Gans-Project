# app.py

import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load generator model from checkpoint
@st.cache_resource
def load_generator():
    generator = make_generator_model()
    checkpoint_dir = './training_checkpoints'
    checkpoint = tf.train.Checkpoint(generator=generator)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    checkpoint.restore(latest).expect_partial()
    return generator

# Generator model definition (must match your training file)
from tensorflow.keras import layers

def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((7, 7, 256)),

        layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# Streamlit App UI
st.title("🧠 Fashion Image Generator using GAN")
st.markdown("Generate synthetic fashion images using a trained DCGAN model.")

generator = load_generator()

if st.button("Generate Fashion Images"):
    noise_dim = 100
    num_images = 16
    seed = tf.random.normal([num_images, noise_dim])
    predictions = generator(seed, training=False)

    st.subheader("Generated Results:")
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        ax.axis('off')
    st.pyplot(fig)
