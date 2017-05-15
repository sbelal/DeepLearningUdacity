import tensorflow as tf;
print(tf.__version__)


node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
result = sess.run([node1, node2])
print(result)

node3 = tf.add(node1, node2)
print("sess.run(node3): ",sess.run(node3))





a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
result = sess.run(adder_node, feed_dict={a:1.0, b:3.4})
print(result)




testInput = [[1, 2, 3, 4], [1, 4, 3,7], [4, 2, 5, 8]]
a1 = tf.placeholder(tf.int32,[None,None])
embeddings = tf.Variable(tf.random_uniform([30, 5]))
embedLayer = tf.nn.embedding_lookup(embeddings, a1)
init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(embedLayer, feed_dict={a1:testInput}))








W = tf.Variable([.3], tf.float32)
b = tf.Variable([-.3], tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
sess.run(init)

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})

print(sess.run([W, b]))







#testInput = [[5, 14, 26, 26, 4], [22, 6, 17], [14, 19, 28, 27, 11]]
#print(testInput[0])

#input = tf.placeholder(tf.int32,shape=[None,None],name='input')









#x = tf.Variable([[[1, 1, 1], [2, 2, 2]],
#             [[3, 3, 3], [4, 4, 4]],
#             [[5, 5, 5], [6, 6, 6]]]
#, tf.float32)


#output = tf.strided_slice(x, [1, 0, 0], [2, 1, 3], [1, 1, 1])

##sess = tf.Session()
##print(sess.run([node1, node2]))

#sess = tf.Session()
#init = tf.global_variables_initializer()
#y = sess.run(init)
#print(sess.run(output))