import tensorflow as tf 
import numpy as np

ModelCheckpoint = tf.keras.callbacks.ModelCheckpoint
def cnn():
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2,
	padding='same', activation='relu', input_shape=(128,128,3))) 
	model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=2,
	padding='same', activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
	model.add(tf.keras.layers.Dropout(0.25))

	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(128,activation='relu'))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(4,activation='softmax'))
	return model

def autoencoder():
	input_img = tf.keras.layers.Input(shape=(128,128,3))
	x = tf.keras.layers.Conv2D(filters=64,kernel_size=3, activation='relu', padding='same')(input_img)
	x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
	x = tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu', padding='same')(x)
	x = tf.keras.layers.MaxPooling2D(pool_size=2)(x)
	x = tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu', padding='same')(x)
	encoded = tf.keras.layers.MaxPooling2D(pool_size=2)(x)

	x = tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', padding='same')(encoded)
	x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
	x = tf.keras.layers.Conv2D(filters=32,kernel_size=3, activation='relu', padding='same')(x)
	x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
	x = tf.keras.layers.Conv2D(filters=64,kernel_size=3, activation='relu', padding='same')(x)
	x = tf.keras.layers.UpSampling2D(size=(2,2))(x)
	decoded = tf.keras.layers.Conv2D(filters=3,kernel_size=3, activation='sigmoid', padding='same')(x)
	
	return tf.keras.models.Model(input_img,decoded)



if __name__ == "__main__":
	flag = 2
	if flag == 1:

		model = cnn()
		model.compile(loss='categorical_crossentropy',
					 optimizer='adam',
					 metrics=['accuracy'])

		data = np.load('final_data.npy')

		np.random.shuffle(data)

		train_len = int(len(data) * 0.8)

		train_data = data[:train_len]
		test_data = data[train_len:]



		train_x = []
		train_y = []
		test_x = []
		test_y = []

		for row in train_data:	
			train_x.append(row[0])
			train_y.append(row[1])

		for row in test_data:
			test_x.append(row[0])
			test_y.append(row[1])

		train_x = np.array(train_x).astype('float32')/255
		train_y = np.array(train_y)

		test_x = np.array(test_x).astype('float32')/255
		test_y = np.array(test_y)

		train_x = train_x.reshape(-1,128,128,3)
		test_x = test_x.reshape(-1,128,128,3)


		checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
		model.fit(train_x,
				  train_y,
				  batch_size=64,
				  epochs=10,
				  validation_data=(test_x,test_y),
				  callbacks=[checkpointer])
	elif flag == 2:
		ae = autoencoder()
		ae.compile(optimizer='adadelta', loss='binary_crossentropy')
		data = np.load('final_data.npy')

		train_len = int(len(data) * 0.8)
		train_data = data[:train_len]
		test_data = data[train_len:]

		train_x = []
		test_x = []
		for row in train_data:
			img = np.array(row[0]).astype('float32') / 255
			train_x.append(img)

		for row in test_data:
			img = np.array(row[0]).astype('float32') / 255
			test_x.append(img)

		train_x = np.array(train_x).reshape(len(train_x),128,128,3)
		test_x = np.array(test_x).reshape(len(test_x),128,128,3)

		ae.fit(train_x,train_x,
			   epochs=10,
			   batch_size=64,
			   shuffle=True,
			   validation_data=(test_x,test_x))
