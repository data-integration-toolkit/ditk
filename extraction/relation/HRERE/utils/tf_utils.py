import tensorflow as tf

def l2_loss(t, axis=-1):
	return tf.sqrt(tf.reduce_sum(tf.square(t), axis=axis))

def l1_loss(t, axis=-1):
	return tf.reduce_sum(tf.abs(t), axis=axis)
