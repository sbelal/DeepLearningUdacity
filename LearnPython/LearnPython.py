import tensorflow as tf;
print(tf.__version__)



input = tf.placeholder(tf.int32, shape=[None,None],name="input")
processed = tf.strided_slice(input,begin=[0,0],end=[3,-1],strides=[1,1])
processed = tf.concat([tf.fill([3, 1], 0), processed], 1)


testInput = [[1, 2, 3, 4, 5], [11, 22, 33, 44,55], [2, 4, 8,16,32]]

sess = tf.Session()
print(sess.run(processed, feed_dict={input:testInput}))



