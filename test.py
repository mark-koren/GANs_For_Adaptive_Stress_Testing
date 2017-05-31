import io
import matplotlib.pyplot as plt
import tensorflow as tf
import pdb
import numpy as np

class GeneratorDistribution(object):
    def __init__(self, range):
        self.range = range

    def gen_plot(self, z):
        """Create a pyplot plot and save to buffer."""
        plt.figure()
        plt.plot([z, 2])
        plt.title("test")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def plots(self, z):
        im = self.gen_plot(z[0])
        for i in range(1, z.shape[0]):
            im = tf.concat([im, self.gen_plot(z[i])], 0)

        return im
# Prepare the plot

    def model(self):
        z = tf.placeholder(tf.float32, shape=(), name='z')
        im_t  = tf.placeholder(tf.float32, shape=[self.range,480,640,4], name='img_tensor')
        # im_t1  = tf.placeholder(tf.float32, shape=[1,480,640,4], name='img_tensor')
        # im_t2  = tf.placeholder(tf.float32, shape=[1,480,640,4], name='img_tensor')
        # summary_op = tf.summary.image("plot", tf.concat([im_t, im_t1, im_t2], 0), max_outputs=3)
        summary_op = tf.summary.image("plot", im_t, max_outputs=self.range)
        # Session
        with tf.Session() as sess:
            # Run
            writer = tf.summary.FileWriter('./test/run6', sess.graph)
            z1 = np.zeros((self.range,))
            for step in range(self.range):
                z1[step] = sess.run(z, feed_dict={z: 4+step})

            summary = sess.run(summary_op, feed_dict={
                im_t: self.plots(z1).eval()})
            writer.add_summary(summary)
            # plot_buf = gen_plot(z)
            # pdb.set_trace()
            # Convert PNG buffer to TF image
            # image = tf.image.decode_png(plot_buf.getvalue(), channels=4)
            #
            # # Add the batch dimension
            # image = tf.expand_dims(image, 0)

            # Add image summary

            pdb.set_trace()
            # Write summary


            writer.close()

if __name__ == '__main__':
    a = GeneratorDistribution(3)
    a.model()