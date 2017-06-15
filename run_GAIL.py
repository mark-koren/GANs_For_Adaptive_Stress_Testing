from Simple_GAIL import *
import numpy as np


num_steps = 4999
batch_size = [1]
minibatch = True
log_every = 100
image_every = 1000
anim_path = None
balance = 0.0
writer_path = './GAIL/dragan5'
learning_rate = [1.0e-2, 1.0e-3, 1.0e-4, 1.0e-5, 1.0e-6]
sample_size = [10]
c_list = [0.5]

for l in learning_rate:
    for b in batch_size:
        for s in sample_size:
            for c in c_list:
                model = GAN(
                    data = DataDistribution(),
                    gen = GeneratorDistribution(low=np.array([0.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0])),
                    num_steps=num_steps,
                    batch_size=b,
                    minibatch=minibatch,
                    log_every=log_every,
                    image_every=image_every,
                    anim_path=anim_path,
                    balance=balance,
                    writer_path = writer_path,
                    learning_rate = l,
                    sample_size= s,
                    C = c
                )
                model.train()