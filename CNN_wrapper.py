import tensorflow as tf
import numpy as np

class CNN:
	def  __init__(self, inp_size, n_classes):
		self.n_classes = n_classes
		self.inp_size = inp_size

		self.epoch_n = 1		
		self.filter_sizes = [10,10]
		self.batch_size = 4
		self.lr = 0.01
		self.global_iteration = 0

	def create_placeholders(self,):	
		self.x = tf.placeholder(dtype=tf.float32, shape=[None,self.inp_size[0],self.inp_size[1],1 ])
		self.y = tf.placeholder(dtype=tf.int64, shape=[None,1])
				

	def build_graph(self):
		#Layer1: Convolution
		#Layer2: ReLU
		#Layer3: Maxpool
		pooling_outputs = []
		num_filters = 2#filter of particular shape
		for i,filter_size in enumerate(self.filter_sizes):
			filter_shape = [filter_size, filter_size, 1, num_filters]
			W = tf.Variable(tf.random_uniform(shape=filter_shape))
			B = tf.Variable(tf.random_uniform(shape=[num_filters]))
			
			conv_output = tf.nn.bias_add(tf.nn.conv2d(self.x,W, strides=[1, 1, 1, 1],padding="VALID"), B)
			
			non_linearity_output = tf.nn.relu(conv_output)
			
			pool_output = tf.nn.max_pool(non_linearity_output, ksize = [1,self.inp_size[0]-filter_size+1,1,1],strides=[1, 1, 1, 1],padding='VALID')
			pooling_outputs.append(pool_output)
		self.pool_outputs = tf.stack(pooling_outputs)
		shp = tf.shape(self.pool_outputs)
		
		#Converting to shape : [batch_size, total_filters]
		self.pool_outputs = tf.reshape(tf.transpose(self.pool_outputs, (1,0,2,3,4)) ,[shp[1],shp[0]*shp[4]*shp[2]*shp[3] ])		

		#Layer5: Fully-Connected Layer	
		#total_filters = shp[0]*shp[4]*shp[2]*shp[3]
		w = tf.Variable(tf.random_uniform(shape=[1964,self.n_classes]))
		b = tf.Variable(tf.random_uniform(shape=[self.n_classes]))
		self.score = tf.nn.xw_plus_b(self.pool_outputs, w, b)
		#self.score = tf.add( tf.nn.softmax( tf.nn.xw_plus_b(self.pool_outputs, w, b) )  , tf.constant(1.0) )
		#self.score = tf.multiply(self.score , tf.constant(self.n_classes*1.0) )

		#Prediction:
		
		
		#Loss & Optimisation:
		#self.loss_train = tf.reduce_mean( tf.square(self.score - self.y) )
		self.loss_train = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.y,27),logits=self.score) )	
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_train)

		#Accuracy
		matches = tf.equal(tf.argmax(self.score,1),self.y)
		self.acc = tf.reduce_mean(tf.cast(matches,tf.float32))*100

	def get_batches(self, inp):
			batch_size = self.batch_size
			x, y = inp 
			n_batches = len(x)//batch_size
			x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
			for ii in range(0, len(x), batch_size):
				yield x[ii:ii+batch_size], y[ii:ii+batch_size]

	def train_batch(self, sess, train_set_batch):        
			batchX, batchY = train_set_batch
			batchY = np.reshape(batchY, [-1,1])
			feed_dict = {self.x:batchX, self.y:batchY}

			acc, sc_, train_loss ,_ = sess.run([self.acc, self.score, self.loss_train, self.optimizer], feed_dict)
			
			return acc, sc_, train_loss
	
	def train(self, train_set):
		self.create_placeholders()
		self.build_graph()       

		sess = tf.Session()
		sess.run(tf.global_variables_initializer() )

		print('Start Training...')
		for epoch_i in range(self.epoch_n):
			for iteration,train_set_batch in enumerate(self.get_batches(train_set), 1):
				 acc, sc_, train_loss = self.train_batch(sess, train_set_batch)

				 # Print Results:
				 if(iteration%1==0):
				     print('Accuracy',acc)
				     print("Scores:  ",sc_)
				     print("Iteration:  ",iteration)
				     print("Train Loss: ",train_loss)
				     print 
				 self.global_iteration+=1

		print('Training Completed...')
		print('--------------------------')

		self.sess = sess

	def test_batch(self, sess, test_set_batch):        
			batchX, batchY = train_set_batch
			batchY = np.reshape(batchY, [-1,1])
			feed_dict = {self.x:batchX, self.y:batchY}

			acc = sess.run([self.acc], feed_dict)
			
			return acc

	def test(self, test_set):
		sess = self.sess
		print('Start Testing...')
		accuracies = []
		for epoch_i in range(self.epoch_n):
			for iteration,test_set_batch in enumerate(self.get_batches(test_set), 1):
				 acc = self.test_batch(sess, test_set_batch)
				 accuracies.append(acc)

		print('Total Accuracy: ',sum(accuracies)/len(accuracies) )

		print('---------------------------')


