import tensorflow as tf
import numpy as np
import os
train_data = np.load('data/training_data_balanced.npy')

WIDTH = 96
HEIGHT = 96
batch_size = 1

x = np.array(list(i[0] for i in train_data)).reshape(-1,WIDTH,HEIGHT,3)
y = np.array(list(i[1] for i in train_data))

train_x = x[:int(len(x)*0.8)]
train_y = y[:int(len(x)*0.8)]

test_x = x[int(len(x)*0.8):]
test_y = y[int(len(x)*0.8):]




filter1 = tf.Variable(tf.random_normal([5,5,3,64]))
filter2 = tf.Variable(tf.random_normal([10,10,64,64]))


X = tf.placeholder(tf.float32,[None,96,96,3], name='X')
Y = tf.placeholder(tf.float32,[None,3], name='Y')


conv1 = tf.nn.conv2d(X,filter1,strides=[1,2,2,1],padding='SAME')
conv1_act = tf.nn.relu(conv1)
max_pool1 = tf.nn.max_pool(conv1_act, ksize=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
conv2 = tf.nn.conv2d(max_pool1,filter2,strides=[1,2,2,1],padding='VALID')
conv2_act = tf.nn.relu(conv2)
max_pool2 = tf.layers.max_pooling2d(conv2_act,pool_size=[2,2],strides=2)
fc_1 = tf.layers.dense(max_pool2,4096)
fc_1_act = tf.nn.relu(fc_1)
fc_2 = tf.layers.dense(fc_1_act,4096)
fc_2_act = tf.nn.relu(fc_2)
flatten = tf.reshape(fc_2_act,[-1,4096])
output = tf.layers.dense(flatten,3)
logits = tf.nn.softmax(output)


print("conv1--------------------------------")
print(conv1.get_shape())
print("conv1_act--------------------------------")
print(conv1.get_shape())
print("max_pool1--------------------------------")
print(max_pool1.get_shape())
print("conv2--------------------------------")
print(conv2.get_shape())
print("conv2_act--------------------------------")
print(conv2_act.get_shape())
print("mp2--------------------------------")
print(max_pool2.get_shape())
print("fc1--------------------------------")
print(fc_1.get_shape())
print("fc1_act--------------------------------")
print(fc_1_act.get_shape())
print("fc2--------------------------------")
print(fc_2.get_shape())
print("fc2_act--------------------------------")
print(fc_2_act.get_shape())
print("flatten--------------------------------")
print(flatten.get_shape())
print("output--------------------------------")
print(output.get_shape())

loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(loss)

saver = tf.train.Saver(max_to_keep=10)

save_dir = 'checkpoints/'

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)



with tf.Session() as sess:
	tf.initialize_all_variables().run()
	for epoch in range(100):
		print(len(train_x))
		total_batch = int(len(train_x)/batch_size)

		for i in range(0,total_batch-1):

			batch_x = train_x[i*batch_size:(i+1)*batch_size]
			batch_y = train_y[i*batch_size:(i+1)*batch_size]

			sess.run(optimizer, feed_dict={X:batch_x, Y:batch_y})

		if epoch % 1 == 0:

			c = sess.run(loss, feed_dict = {X:train_x, Y:train_y})
			print("Epoch: {}, cost : {:.9f}".format(epoch+1,c))
		saver.save(sess=sess, save_path=get_save_path(1))
