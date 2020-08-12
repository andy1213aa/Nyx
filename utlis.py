import tensorflow as tf


def generator_loss(fake_logit, real_data, fake_data_by_real_parameter):
        l2_norm = tf.norm(tensor = (fake_data_by_real_parameter-real_data[1]), ord='euclidean')/(16*16*16)
        g_loss = - tf.reduce_mean(fake_logit)
        return g_loss, l2_norm

def discriminator_loss(real_logit, fake_logit):
        real_loss = -tf.reduce_mean(real_logit)
        fake_loss = tf.reduce_mean(fake_logit)
        return real_loss, fake_loss

def gradient_penality(dis, real_data, fake_data):
    def _interpolate(a, b):
        shape = [tf.shape(a)[0]]+[1]*(a.shape.ndims - 1)
        alpha = tf.random.uniform(shape = shape, minval = 0, maxval = 1.)
        inter = (alpha * a) + ((1-alpha)*b)
        inter.set_shape(a.shape)
        return inter
    x_img = _interpolate(real_data[1], fake_data)
    with tf.GradientTape() as tape:
        tape.watch(x_img)
        pred_logit = dis(real_data[0][0], real_data[0][1], real_data[0][2], x_img)
        #pred_logit = dis([x_img])
    grad = tape.gradient(pred_logit, x_img)
    norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis = 1)
    gp_loss = tf.reduce_mean((norm-1.)**2)
    return gp_loss    
    