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
    model.add(tf.keras.layers.Dense(3,activation='softmax'))
    return model





if __name__ == "__main__":

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
		if row[1] != [0,0,0]:
			train_x.append(row[0])
			train_y.append(row[1])

	for row in test_data:
		if row[1] != [0,0,0]:
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
			  epochs=2,
			  validation_data=(test_x,test_y),
			  callbacks=[checkpointer])
