from Simple_GAIL import *
import numpy as np


num_steps = 14999
batch_size = 4
minibatch = True
log_every = 100
image_every = 1000
anim_path = None
balance = 0.0
writer_path = './GAIL/playground6'
learning_rate = 6.0e-3
sample_size = 100

# for l in learning_rate:
#     for b in batch_size:
#         for s in sample_size:
#             pass

model = GAN(
    data = DataDistribution(),
    gen = GeneratorDistribution(low=np.array([0.0, 0.0, 0.0]), high=np.array([4.0, 9.0, 99.0])),
    num_steps=num_steps,
    batch_size=batch_size,
    minibatch=minibatch,
    log_every=log_every,
    image_every=image_every,
    anim_path=anim_path,
    balance=balance,
    writer_path = writer_path,
    learning_rate = learning_rate,
    sample_size= sample_size
)
model.train()