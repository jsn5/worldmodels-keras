import tensorflow as tf
from tensorflow.contrib.keras import backend as K
import numpy as np
import cv2

latent_dim = 2
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


input_img = tf.keras.layers.Input(shape=(128,128,3))
x = tf.keras.layers.Conv2D(filters=128,kernel_size=3, activation='relu', padding='same')(input_img)
x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
x = tf.keras.layers.Conv2D(filters=64,kernel_size=3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
x = tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu', padding='same')(x)
x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
shape = K.int_shape(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(16)(x)

z_mean = tf.keras.layers.Dense(latent_dim)(x)
z_log_var = tf.keras.layers.Dense(latent_dim)(x)
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name="z")([z_mean,z_log_var])

encoder = tf.keras.models.Model(input_img, [z_mean, z_log_var,z], name="encoder")
encoder.summary()


latent_inputs = tf.keras.layers.Input(shape=(latent_dim,), name='z_sampling')
x = tf.keras.layers.Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
x = tf.keras.layers.Reshape((shape[1],shape[2],shape[3]))(x)
x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
x = tf.keras.layers.Conv2D(filters=64,kernel_size=3, activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
x = tf.keras.layers.Conv2D(filters=128,kernel_size=3, activation='relu', padding='same')(x)
x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
x = tf.keras.layers.Conv2D(filters=3,kernel_size=3, activation='sigmoid', padding='same')(x)

decoder = tf.keras.models.Model(latent_inputs,x,name='decoder')

decoder.summary()


outputs = decoder(encoder(input_img)[2])
vae = tf.keras.models.Model(input_img,outputs,name="vae")

if __name__ == '__main__':

	data = np.load("final_data.npy")

	train_data =  data[: int(len(data) * 0.8)]

	test_data = data[int(len(data) * 0.8):]

	train_x = []
	test_x = []

	for row in train_data:
		train_x.append(row[0])

	for row in test_data:
		test_x.append(row[0])
	train_x = np.array(train_x).reshape(-1,128,128,3)
	test_x = np.array(test_x).reshape(-1,128,128,3)
	reconstruction_loss = tf.keras.losses.binary_crossentropy(K.flatten(input_img),
															  K.flatten(outputs))

	reconstruction_loss *= 128 * 128

	kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
	kl_loss = K.sum(kl_loss,axis=-1)
	kl_loss *= -0.5
	vae_loss = K.mean(reconstruction_loss + kl_loss)
	vae.add_loss(vae_loss)
	vae.compile(optimizer='rmsprop')
	vae.summary()
	flag = 1

	if flag == 0:
		vae.fit(train_x,
				epochs=10,
				batch_size=64,
				validation_data=(test_x,None))
		vae.save_weights("vae_cnn.h5")
	elif flag == 1:
		vae.load_weights('vae_cnn.h5')
		image = cv2.imread("input.jpg")
		while True:
			image = cv2.resize(image,(128,128))
			image = np.array(image).astype("float32") / 255
			image = image.reshape(1,128,128,3)
			output = vae.predict(image)
			output = np.array(output).reshape(128,128,3)
			output = output * 255
			output = np.array(output).astype("uint8")
			image = output
			cv2.imshow("test",output)
			cv2.waitKey(0)
