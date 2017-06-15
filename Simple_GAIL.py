import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from matplotlib import animation
import argparse
import io
import gym
from ops import *


import pdb


sns.set(color_codes=True)

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


class DataDistribution(object):
    def __init__(self):
        self.mu = np.array([3.0, 1.0])
        self.sigma = np.array([[1.0, 0.0],
                              [0.0, 1.0]])

    def sample(self, N):
        x_samples, y_samples = np.random.multivariate_normal(self.mu, self.sigma, N).T
        x_samples.sort()
        y_samples.sort()
        return x_samples, y_samples

def pertubate_batch(batch, C):
    return batch + C * np.std(batch, axis=0) * np.random.random(batch.shape)



class GeneratorDistribution(object):
    def __init__(self, low, high):
        self.low = low
        self.high = high

    def sample(self, N):
        samples = np.zeros((N, 3))
        samples[:,2] = np.random.uniform(self.low[2], self.high[2], N)
        samples[:,1] = np.random.uniform(self.low[1], self.high[1], N)
        samples[:,0] = np.random.randint(low = self.high[0]+1,size= N)
        return samples

def sample_policy(g, sample_size, state):
    dist = tf.contrib.distributions.MultivariateNormalDiag(mu = tf.slice(g, begin=[0,0], size=[-1,2]),
                                                           diag_stdev = tf.clip_by_value(
                                                               tf.exp(tf.slice(g, begin=[0,2], size=[-1,2])), 0.0, 1e15))

    return tf.concat([tf.reshape(dist.sample((sample_size)), (-1,2)),
                     state],
                     1)


def linear(input, output_dim, scope=None, stddev=1.0):
    norm = tf.random_normal_initializer(stddev=stddev)
    const = tf.constant_initializer(0.0)
    with tf.variable_scope(scope or 'linear'):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, h_dim, training=True):
    # h0 = tf.nn.relu(linear(input, h_dim, 'g0'))
    # h1 = tf.tanh(linear(h0, 1, 'g1'))
    # h0 = tf.layers.dropout(
    #     tf.nn.softplus(
    #         tf.contrib.layers.batch_norm(linear(input, h_dim, 'g0'),
    #                                      center=True, scale=True, is_training=True)),
    #     training=training)
    # h0 = tf.layers.dropout(lrelu(bn(linear(input, h_dim, 'g0'), phase=True, name='g0_bn')), training=True)
    # h1 = tf.layers.dropout(lrelu(bn(linear(h0, h_dim, 'g1'), phase=True, name='g1_bn')), training=True)
    h0 = tf.nn.dropout(lrelu(bn(linear(input, h_dim*32, 'g0'), phase=True, name='g0_bn')), keep_prob=0.5)
    h1 = lrelu(bn(linear(h0, h_dim*16, 'g1'), phase=True, name='g1_bn'))
    h2 = lrelu(bn(linear(h1, h_dim * 8, 'g2'), phase=True, name='g2_bn'))
    h3 = lrelu(bn(linear(h2, h_dim * 4, 'g3'), phase=True, name='g3_bn'))
    # h0 = tf.nn.softplus(linear(input, h_dim, 'g0'))
    h4 = linear(h3, 4, 'g4')
    return h4


def discriminator(input, h_dim, minibatch_layer=True):
    # h0 = tf.tanh(linear(input, h_dim * 2, 'd0'))
    # h1 = tf.tanh(linear(h0, h_dim * 2, 'd1'))

    # without the minibatch layer, the discriminator needs an additional layer
    # to have enough capacity to separate the two distributions correctly
    # if minibatch_layer:
    #     h2 = minibatch(h1)
    # else:
    #     h2 = tf.tanh(linear(h1, h_dim * 2, scope='d2'))
    h0 = lrelu(bn(linear(input, h_dim * 8, 'd0'), phase=True, name='d0_bn'))
    h1 = lrelu(bn(linear(h0, h_dim * 4, 'd1'), phase=True, name='d1_bn'))
    h2 = lrelu(bn(linear(h1, h_dim * 2, 'd2'), phase=True, name='d2_bn'))
    h3 = linear(h2, 1, scope='d3')


    # h0 = tf.nn.relu(linear(input, h_dim * 2, 'd0'))
    # h1 = tf.nn.relu(linear(h0, h_dim * 2, 'd1'))
    #
    # # without the minibatch layer, the discriminator needs an additional layer
    # # to have enough capacity to separate the two distributions correctly
    # if minibatch_layer:
    #     h2 = minibatch(h1)
    # else:
    #     h2 = tf.nn.relu(linear(h1, h_dim * 2, scope='d2'))
    #
    # h3 = tf.nn.relu(linear(h2, 1, scope='d3'))
    return tf.sigmoid(h3), h3


def minibatch(input, num_kernels=5, kernel_dim=3):
    x = linear(input, num_kernels * kernel_dim, scope='minibatch', stddev=0.02)
    activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
    diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
    abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
    minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)
    # pdb.set_trace()
    return tf.concat([input, minibatch_features], 1)


def optimizer(loss, var_list, initial_learning_rate):
    decay = 0.99
    num_decay_steps = 1000
    batch = tf.Variable(0)
    learning_rate = tf.train.exponential_decay(
        initial_learning_rate,
        batch,
        num_decay_steps,
        decay,
        staircase=True
    )
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(
    #     loss,
    #     global_step=batch,
    #     var_list=var_list
    # )
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5).minimize(
        loss,
        global_step=batch,
        var_list=var_list
    )
    return optimizer


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, minibatch, log_every, image_every,
                 anim_path, balance = 0.25, writer_path = './GAIL/DRAGAN1', learning_rate = 50.3e-4,
                 sample_size = 100, C=0.1):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.minibatch = minibatch
        self.log_every = log_every
        self.image_every = image_every
        self.num_images = (num_steps // image_every)*2 + 3 #+2 for one at iter 0 and one after all iters
        self.mlp_hidden_size = 8
        self.anim_path = anim_path
        self.anim_frames = []
        self.dist_list = []
        self.writer_path = writer_path
        self.sample_size = sample_size
        self.balance = balance
        self.C = C
        # self.learning_rate = learning_rate
        self.filename = "LR"+str(learning_rate) + \
                        "_NS" + str(self.num_steps) + \
                        "_BS" + str(self.batch_size) +  \
                        "_HS" + str(self.mlp_hidden_size) + \
                        "_SS" + str(self.sample_size) + \
                        "_C" + str(self.C)


        # can use a higher learning rate when not using the minibatch layer
        if self.minibatch:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = 0.03

        self._create_model()

    def _create_model(self):
        tf.reset_default_graph()
        # In order to make sure that the discriminator is providing useful gradient
        # information to the generator from the start, we're going to pretrain the
        # discriminator using a maximum likelihood objective. We define the network
        # for this pretraining step scoped as D_pre.
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 2))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 2))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size, self.minibatch)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        # This defines the generator network - it takes samples from a noise
        # distribution as input, and passes them through an MLP.

        # with tf.variable_scope('G_pre') as scope:
        #     self.pre_z = tf.placeholder(tf.float32, shape=(self.batch_size, 3))
        #     # self.training = tf.placeholder(dtype=tf.bool)
        #     G_pre = generator(self.pre_z, self.mlp_hidden_size, training=True)
        #     self.g_pre_loss = tf.reduce_mean(tf.square(G_pre - 3.0))
        #     self.g_pre_opt = optimizer(self.g_pre_loss, None, self.learning_rate)
        #     scope.reuse_variables()

        with tf.variable_scope('Gen') as scope:
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 3))
            self.training = tf.placeholder(dtype=tf.bool)
            self.G = generator(self.z, self.mlp_hidden_size, training=self.training)
            scope.reuse_variables()
            # self.G2 = self._eval_G2()
            # self.G2 = generator(self.z, self.mlp_hidden_size)


        # The discriminator tries to tell the difference between samples from the
        # true data distribution (self.x) and the generated samples (self.z).
        #
        # Here we create two copies of the discriminator network (that share parameters),
        # as you cannot use the same network with different inputs in TensorFlow.
        with tf.variable_scope('Disc') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size*self.sample_size, 2))
            #X = tf.placeholder(tf.float32, shape=[None, X_dim])
            self.x_p = tf.placeholder(tf.float32, shape=(self.batch_size*self.sample_size, 2))
            self.z_expanded = tf.placeholder(tf.float32, shape=(self.batch_size * self.sample_size, 3))
            self.D1, self.D1_logits = discriminator(tf.concat([self.x, self.z_expanded],1), self.mlp_hidden_size, self.minibatch)
            scope.reuse_variables()
            self.sample = sample_policy(self.G, sample_size=self.sample_size, state=self.z_expanded)
            self.D2, self.D2_logits = discriminator(self.sample,
                                    self.mlp_hidden_size, self.minibatch)
            # Gradient penalty
            alpha = tf.random_uniform(
                shape=[self.batch_size, 1],
                minval=0.,
                maxval=1.
            )
            self.lambd = 10
            differences = self.x_p - self.x
            interpolates = self.x + (alpha * differences)
            # print (type(interpolates))
            # pdb.set_trace()
            self.D3, self.D3_logits = discriminator(tf.concat([interpolates, self.z_expanded],1), self.mlp_hidden_size)
            gradients = tf.gradients(ys=self.D3, xs=[interpolates], name='interp_grad')[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            self.gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)

        # Define the loss for discriminator and generator networks (see the original
        # paper for details), and create optimizers for both
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits, labels=tf.ones_like(self.D1_logits)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.zeros_like(self.D2_logits)))
        self.loss_d = tf.reduce_mean(self.d_loss_real + self.d_loss_fake)
        # self.loss_g = tf.reduce_mean(-tf.log(self.D2))
        self.loss_g_d = tf.reduce_mean(-tf.log(self.D2))
        # self.loss_g_g = tf.reduce_mean(self.balance * tf.log(1.0 + tf.abs(3.0-self.G)))
        self.loss_g = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.ones_like(self.D2_logits)))

        self.loss_d += self.lambd * self.gradient_penalty

        self.image_tensor = tf.placeholder(tf.float32, shape=[self.num_images, 550, 800, 4], name='image_tensor')
        with tf.name_scope('summaries'):
            # sum_g2 = self.G2
            with tf.name_scope('d_losses'):
                d1 = tf.summary.scalar('loss_d', self.loss_d)
            with tf.name_scope('g_losses'):
                sum_loss_g = tf.summary.scalar('loss_g', self.loss_g)
                sum_mean_x = tf.summary.scalar('x_mean', tf.reduce_mean(self.G[:,0]))
                sum_mean_y = tf.summary.scalar('y_mean', tf.reduce_mean(self.G[:, 1]))
                sum_stddev_x = tf.summary.scalar('x_stddev', tf.reduce_mean(tf.exp(self.G[:, 2])))
                sum_stddev_y = tf.summary.scalar('y_stddev', tf.reduce_mean(tf.exp(self.G[:, 3])))
                sum_sample_x = tf.summary.histogram('samples_x', self.sample[:,0])
                sum_sample_y = tf.summary.histogram('samples_y', self.sample[:,1])
                # sum_loss_g_g = tf.summary.scalar('loss_g_g', self.loss_g_g)
                # sum_loss_g_d = tf.summary.scalar('loss_g_d', self.loss_g_d)
                # sum_loss_g_g_percent = tf.summary.scalar('percent_g', self.loss_g_g / self.loss_g)
                # sum_loss_g_d_percent = tf.summary.scalar('percent_d', self.loss_g_d / self.loss_g)
                # sum_loss_ratio = tf.summary.scalar('loss_ratio', self.loss_g_g / self.loss_g_d)
                # sum_loss_ratio_no_coeff = tf.summary.scalar('loss_ratio_no_coeff', (self.loss_g_g / max(self.balance, 1e-7)) / self.loss_g_d)

            with tf.name_scope('images'):
                self.dists = tf.summary.image("plot", self.image_tensor, max_outputs=self.num_images)
            # with tf.name_scope('images'):
            #     # plot_buf = self._save_distributions()
            #     image = tf.image.decode_png(self.plot_buf.getvalue(), channels=4)
            #     image = tf.expand_dims(image, 0)
            #     image_sum = tf.summary.image('image', image)

        self.d_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='D_pre')
        self.g_pre_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='G_pre')
        self.d_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Disc')
        self.g_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Gen')

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate/100)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)
        self.g_summary = tf.summary.merge([sum_loss_g,
                                           sum_mean_x,
                                           sum_mean_y,
                                           sum_stddev_x,
                                           sum_stddev_y,
                                           sum_sample_x,
                                           sum_sample_y])
        self.d_summary = tf.summary.merge([d1])


    def train(self):


        with tf.Session() as session:
            # merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(self.writer_path + "/" + self.filename, session.graph)
            tf.global_variables_initializer().run()
            #
            # # pretraining discriminator
            # if self.balance >= 1.0:
            #     num_pretrain_steps = 0
            # else:
            #     num_pretrain_steps = 0
            # for step in range(num_pretrain_steps):
            #     d = (np.random.random(self.batch_size) - 0.5) * 10.0
            #     labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
            #     pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
            #         self.pre_input: np.reshape(d, (self.batch_size, 1)),
            #         self.pre_labels: np.reshape(labels, (self.batch_size, 1))
            #     })
            # self.weightsD = session.run(self.d_pre_params)
            #
            # # copy weights from pre-training over to new D network
            # for i, v in enumerate(self.d_params):
            #     session.run(v.assign(self.weightsD[i]))

            # pretraining generator
            # if self.balance >= 1.0:
            #     num_pretrain_steps = 100
            # else:
            #     num_pretrain_steps = 5
            # num_pretrain_steps_g = 300
            # for step in range(num_pretrain_steps_g):
            #     z = self.gen.sample(self.batch_size)
            #     # labels = norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
            #     pretrain_loss, _ = session.run([self.g_pre_loss, self.g_pre_opt], {
            #         self.pre_z: np.reshape(z, (self.batch_size, 1))
            #     })
            # self.weightsG = session.run(self.g_pre_params)

            # copy weights from pre-training over to new D network
            # for i, v in enumerate(self.g_params):
            #     session.run(v.assign(self.weightsG[i]))

            for step in range(self.num_steps):
                # update discriminator
                x = self.data.sample(self.batch_size*self.sample_size)
                z = self.gen.sample(self.batch_size)
                z_expanded = np.repeat(z, self.sample_size, axis=0)
                loss_d, _, d_summary = session.run([self.loss_d, self.opt_d, self.d_summary], {
                    self.x: np.reshape(x, (self.batch_size*self.sample_size, 2)),
                    self.x_p: pertubate_batch(np.reshape(x, (self.batch_size*self.sample_size, 2)), self.C),
                    self.z: np.reshape(z, (self.batch_size, 3)),
                    self.z_expanded: np.reshape(z_expanded, (self.batch_size*self.sample_size, 3)),
                    self.training: True
                })
                train_writer.add_summary(d_summary, step)
                # update generator
                z = self.gen.sample(self.batch_size)
                z_expanded = np.repeat(z, self.sample_size, axis=0)
                # z_eval = np.linspace(-self.gen.range, self.gen.range, 10000)
                loss_g, _, g_value, g_summary = session.run([self.loss_g, self.opt_g, self.G, self.g_summary], {
                    self.z: np.reshape(z, (self.batch_size, 3)),
                    self.z_expanded: np.reshape(z_expanded, (self.batch_size * self.sample_size, 3)),
                    self.training: True
                })
                # update generator
                z = self.gen.sample(self.batch_size)
                z_expanded = np.repeat(z, self.sample_size, axis=0)
                # z_eval = np.linspace(-self.gen.range, self.gen.range, 10000)
                loss_g, _, g_value, g_summary = session.run([self.loss_g, self.opt_g, self.G, self.g_summary], {
                    self.z: np.reshape(z, (self.batch_size, 3)),
                    self.z_expanded: np.reshape(z_expanded, (self.batch_size * self.sample_size, 3)),
                    self.training: True
                })
                train_writer.add_summary(g_summary, step)

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))

                    # print(g_value)
                    # pdb.set_trace()

                # if step % self.image_every == 0:
                #     self.dist_list.append(self._samples(session, training=True))
                #     self.dist_list.append(self._samples(session, training=False))
                    # image = self._save_distributions(session)
                    # train_writer.add_summary(image)

                # if self.anim_path:
                #     self.anim_frames.append(self._samples(session))

            # if self.anim_path:
            #     self._save_animation()
            # else:
                # self._plot_distributions(session)
                # self.dist_list.append(self._samples(session))
                # image_summary = self._save_distributions(session)
                # train_writer.add_summary(image_summary)
                # train_writer.close()

    def _samples(self, session, num_points=10000, num_bins=100, training = True):
        """
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        """
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            db[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.D1, {
                self.x: np.reshape(
                    xs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1),
                ),
                self.training: training
            })

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                ),
                self.training: training
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return (db, pd, pg)

    def _plot_distributions(self, session, dists):
        db, pd, pg = dists
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        return plt


    def _show_distributions(self, session):
        for i in range(len(self.dist_list)):
            plt = self._plot_distributions(session, self.dist_list[i])
            plt.show()

    def _image_distributions(self, session, dists):
        plt = self._plot_distributions(session, dists)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        image = tf.expand_dims(image, 0)
        return image

    def _save_distributions(self, session):
        im = self._image_distributions(session, self.dist_list[0])
        for i in range(1, len(self.dist_list)):
            im = tf.concat([im, self._image_distributions(session, self.dist_list[i])], 0)
        summary = session.run(self.dists, feed_dict={
            self.image_tensor: im.eval()})
        return summary


    def _save_animation(self):
        f, ax = plt.subplots(figsize=(6, 4))
        f.suptitle('1D Generative Adversarial Network', fontsize=15)
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 1.4)
        line_db, = ax.plot([], [], label='decision boundary')
        line_pd, = ax.plot([], [], label='real data')
        line_pg, = ax.plot([], [], label='generated data')
        frame_number = ax.text(
            0.02,
            0.95,
            '',
            horizontalalignment='left',
            verticalalignment='top',
            transform=ax.transAxes
        )
        ax.legend()

        db, pd, _ = self.anim_frames[0]
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))

        def init():
            line_db.set_data([], [])
            line_pd.set_data([], [])
            line_pg.set_data([], [])
            frame_number.set_text('')
            return (line_db, line_pd, line_pg, frame_number)

        def animate(i):
            frame_number.set_text(
                'Frame: {}/{}'.format(i, len(self.anim_frames))
            )
            db, pd, pg = self.anim_frames[i]
            line_db.set_data(db_x, db)
            line_pd.set_data(p_x, pd)
            line_pg.set_data(p_x, pg)
            return (line_db, line_pd, line_pg, frame_number)

        anim = animation.FuncAnimation(
            f,
            animate,
            init_func=init,
            frames=len(self.anim_frames),
            blit=True
        )
        anim.save(self.anim_path, fps=30, extra_args=['-vcodec', 'libx264'])


def main(args):
    model = GAN(
        DataDistribution(),
        GeneratorDistribution(low = np.array([0.0,0.0,0.0]), high = np.array([4.0, 9.0, 99.0])),
        args.num_steps,
        args.batch_size,
        args.minibatch,
        args.log_every,
        args.image_every,
        args.anim,
        args.balance
    )
    model.train()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-steps', type=int, default=4999,
                        help='the number of training steps to take')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='the batch size')
    parser.add_argument('--minibatch', type=bool, default=True,
                        help='use minibatch discrimination')
    parser.add_argument('--log-every', type=int, default=100,
                        help='print loss after this many steps')
    parser.add_argument('--image-every', type=int, default=1000,
                        help='Save plots of distributions to tensorboard after this many steps')
    parser.add_argument('--anim', type=str, default=None,
                        help='name of the output animation file (default: none)')
    parser.add_argument('--balance', type=float, default=0.25,
                        help='Balance Hyperparameter of G and D losses')
    # parser.add_argument('--anim', type=str, default='./anim.mp4',
    #                     help='name of the output animation file (default: none)')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())

