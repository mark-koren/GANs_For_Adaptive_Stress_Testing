"""
Tristan McRae
AA290
Winter 2017

This program trains a neural net to predict the final cost a drive will 
result in based on the current state of the system and the action taken by a
pedestrian.
"""

"""Outside libraries used. Tensorflow is used for learning, pandas is used 
for reading CSVs and numpy is used for data manupulation"""
import tensorflow as tf
import pandas
import numpy as np

FLAGS = None

"""Defines a standard weight variable to simplify creation of the net later.
Mean is set to .5 because it gives the best imperical results. Mean should be
positive to prevent the ReLU function employed later in the program from 
deacivating a lot of neurons. stdev should be non-zero to introduce noise
which can often help convergence in neural nets. The datatype is set to 
float64 (not float32) because the initial errors can be very high and 
we don't want to run out of room to store those numbers."""
def weight_variable(shape):
  initial = tf.truncated_normal(shape, mean=.5, stddev=0.1, dtype = tf.float64)  
  return tf.Variable(initial)

"""Defines a standard bias variable to simplify creation of the net later. Like
with weights, this should be initialized to a positive to avoid having neurons
immediately deactivated by the ReLU."""
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape, dtype = tf.float64) #keeps ReLU alive to start
  return tf.Variable(initial)


def main():
  """Imports data from CSV format from a specified directory and converts 
  it to a matrix. This data should be generated by Julia code. Two datasets
  are imported, one for training and the other for testing. Each dataset
  has one file containing state and action information (inputs) while the
  other containts corresponding cost information (labels) """
  print "running now"
  print "reading in data"
  inputs_test = pandas.read_csv("random_baseline_test_inputs.csv").values
  labels_test = pandas.read_csv("random_baseline_test_labels.csv").values
  inputs_train = pandas.read_csv("two_layer_generated2_train_inputs.csv").values
  labels_train = pandas.read_csv("two_layer_generated2_train_labels.csv").values

  #Keeps track of the number of samples in each dataset.
  train_size = len(labels_train)
  test_size = len(labels_test)

  """Batch size is how many samples to train/evaluate the model on at once.
  In my experience, this doesn't effect performance of the model but if it is
  too high, your machine can run out of memory and crash and if it is too low
  your training can take too long. 100 has worked well so far."""
  batch_size=100

  """The model will train for a number of epochs equal to num_epochs. One epoch
  passes when the model has looked at every point in the dataset for training.
  The more epochs, the better trained the model will be and the better performance
  it will have. More complex models tend to take more epochs to train properly. 
  There will be a point of diminishing returns with the epochs. For this model
  improvements start to flatten out around 10 epochs"""
  num_epochs = 10  

  """This is a small constant used in batch regularization calculations to prevent
  divide by zero errors. At present batch regularization code is not implemented
  but is commented out (more on that further down)"""
  delta = 10^-8 

  """These are paramaters for dropout. Dropout is a way of preventing overfitting by
  temporarily removing neurons (by setting them to 0) during training. The idea is 
  that you are forcing the rest of the neurons to compensate for the lost neuron and
  you are preventing neurons from "conspiring" to overfit. drop_input and drop_hidden
  are the proportions (out of 1) of the dropout inputs that are kept when dropout is 
  called with those variables. Standard practice is to keep .8 of the input units and 
  .5 of the units in each hidden layer. Here both keep probabilities are set to 1 meaning
  no dropout occurs. This is because overfitting has not yet become an issue. If it does
  (signified by a large difference between train and test errors), dropout can be 
  implemented as a potential fix by changing these variables."""
  drop_input = 1
  drop_hidden = 1


  print "creating model"
  """ x is a placeholder for the input data that is going to be passed in to the model.
  Later on when we tell the model to start training, we will tell it to look at the 
  data we took from the CSVs to use as values for x. A value of [None, 5] in the shape 
  argument means that there can be an arbitrary amount of length 5 vectors passed in"""
  x = tf.placeholder(tf.float64, [None, 5], name = "input_placeholder")


  """These are placeholders for the dropout keep probabilities. We want dropout to occur only during 
  training and not testing so we make keep probabilities placeholders and pass in the value we want
  later when we know which (training/testing) we are doing."""
  drop_input_placeholder = tf.placeholder(tf.float64, shape = (), name = "drop_input_placeholder")
  drop_hidden_placeholder = tf.placeholder(tf.float64, shape = (), name = "drop_hidden_placeholder")

  #Applies dropout to inputs
  x_drop = tf.nn.dropout(x, drop_input_placeholder)

  """If reuse is set to false, weights and biases are initialized according to defaults. If true,
  it reads in the initial values of weights and biases from saved CSVs"""
  reuse = False

  """This is the first layer of the neural network. This layer accepts [batch_size x 5] input vectors 
  and multiplies them be a bias and adds a weight to create a [batch_size x 10] hidden vector. It then
  applies a ReLU activation function to force all elements of h_1 to be non-negative. This effectively
  either turns off each neuron or turns them on with an intensity linerly proportional to the value of
  the hidden unit. Subsequent layers are shaped similarly."""

  if (reuse):
    W_1 = tf.Variable(pandas.read_csv("two_layer_weights_1.csv", header = None).values)
    b_1 = tf.Variable(pandas.read_csv("two_layer_biases_1.csv", header = None).values)
  else:
    W_1 = weight_variable([5,10])
    b_1 = bias_variable([10])
  h_1 = tf.nn.relu(tf.matmul(x_drop, W_1) + b_1)

  """Batch normalization is a technique that has been shown in other cases to dramatically help convergence.
  The commented out lines here show the form that it would take in this implementation. If one wanted to 
  apply batch normalization to this entire program, they would just have to uncomment the lines below and
  copy them in every step of the neural network. (e.g. change sig_1 to sig_2, mu_1 to mu_2 etc. and put the 
  steps between the calculations of h_2 and h_2_drop.) Imperically this does not help the current model as it
  stands but it should be retried every once in a while if hte model is changing becuase it has been so 
  effective in practice.
  mu_1 = tf.reduce_mean(h_1)
  sig_1 = tf.sqrt(tf.reduce_mean(tf.squared_difference(h_1, mu_1))+delta)
  h_1 = tf.divide((h_1-mu_1),sig_1)"""

  #Applies dropout to the first hidden layer.
  h_1_drop = tf.nn.dropout(h_1, drop_hidden_placeholder)

  #Second hidden layer, takes [batch_size x 10] input and gives [batch_size x 5] output
  if (reuse):
    W_2 = tf.Variable(pandas.read_csv("two_layer_weights_2.csv", header = None).values)
    b_2 = tf.Variable(pandas.read_csv("two_layer_biases_2.csv", header = None).values)
  else: 
    W_2 = weight_variable([10,5])
    b_2 = bias_variable([5])
  h_2 = tf.nn.relu(tf.matmul(h_1_drop, W_2) + b_2)
  h_2_drop = tf.nn.dropout(h_2, drop_hidden_placeholder)


  #[batch_size x 5] -> [batch_size x 1]
  if (reuse):
    W_3 = tf.Variable(pandas.read_csv("two_layer_weights_3.csv", header = None).values)
    b_3 = tf.Variable(pandas.read_csv("two_layer_biases_3.csv", header = None).values)
  else: 
    W_3 = weight_variable([5,1])
    b_3 = bias_variable([1])
  y = tf.matmul(h_2_drop, W_3) + b_3


  print "making definitions"


  # y_ is a placeholder for the actual costs. It will later be filled in with label data.
  y_ = tf.placeholder(tf.float64, [None,], name="label_placeholder")


  """This is the cost for the neural network (not the pedestrian). Higher cost reflects
  predictions being further from reality."""
  cost = tf.squared_difference(y, y_)

  #Accuracy is the average cost for the batch
  accuracy = tf.reduce_mean(cost) 

  """This is where the magic happens. Your optimizer will modify the weight and bias variables
  to minimize this cost as best it can"""
  train_step = tf.train.AdamOptimizer().minimize(cost)

  #Creates the session in which your model will run
  sess = tf.InteractiveSession()

  #In different versions of tensorflow, initialize_all_variables is used instead of global_variables_initializer
  tf.global_variables_initializer().run() 
  
  print "training"

  #Calculating the number of batches you need based on number of samples and batch size
  num_batches = int(np.floor(train_size/batch_size))


  """The way I've set this up, the entire dataset doesn't get looked at, it can only look at up to a 
  multiple of the batch size. There are ways to account for everything but it isn't super critical to
  the effectiveness of the model so I've let it slide for now. These printouts give you a sense of how 
  much data you are ignoring"""
  print "train size"
  print train_size
  print "actual size I'm sampling from"
  print num_batches*batch_size

  for j in range (num_epochs): #This for loop iterated over epochs to train the model
    print "epoch %d of %d" % (j+1, num_epochs)

    """In order to keep consecutive samples independent of each other and allow training to converge better,
    the dataset must be shuffled before being trained. The following lines shuffle the data once per epoch. 
    It is important that inputs and labels are shuffled in the same order so that their rows still correspond
    to each other."""
    permutation = np.random.permutation(len(labels_train))

    print "shuffling"
    shuffled_inputs_train = np.copy(inputs_train)
    shuffled_labels_train = np.copy(labels_train)

    for old_index, new_index in enumerate(permutation):
        shuffled_labels_train[new_index, 0] = np.copy(labels_train[old_index,0])
        shuffled_inputs_train[new_index, 0:5] = np.copy(inputs_train[old_index,0:5])
 
    print "training"

    #Iterates training over every batch    
    batch_error_total = 0 #resets the error counter
    for i in range(num_batches):  
      batch_xs = shuffled_inputs_train[i*batch_size:(i+1)*batch_size,0:5]  #takes states and actions for the current batch
      batch_ys = shuffled_labels_train[i*batch_size:(i+1)*batch_size,0]    #takes labels (costs) for the current batch
      #the line below initiates the training and keeps track of accuracy. The feed_dict shows where to look for placeholder values
      _, batch_error = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_: batch_ys, drop_hidden_placeholder: drop_hidden, drop_input_placeholder: drop_input})
      batch_error_total += float(batch_error)/num_batches #keeps track of error
    avg_train_error = batch_error_total
    print "epoch %d has average training error of %d" % (j+1, avg_train_error)
      

  print "testing"
  """keeps track of the maximum and minimum predictions. This is useful to see how "adventurous" the model gets in
  its predictions. A small range means it isn't really making much different precitions every time while a larger range
  means it is picking up on patterns and is really able to discriminate between high likelihood and low likelihood crash
  scenarios"""
  y_max = tf.reduce_max(y)
  y_min = tf.reduce_min(y)


  #Similar to during training, our testing is batched
  num_batches = int(np.floor(test_size/batch_size))
  print "test size"
  print test_size
  print "actual size I'm sampling from"
  print num_batches*batch_size

  batch_error_total = 0
  ymax = 0
  ymin = 9999999999999999999999999999
  for i in range(num_batches):  
    batch_xs = inputs_test[i*batch_size:(i+1)*batch_size,0:5] 
    batch_ys = labels_test[i*batch_size:(i+1)*batch_size,0]
    batch_error, y_max_, y_min_ = sess.run([accuracy, y_max, y_min], feed_dict={x: batch_xs, y_: batch_ys, drop_input_placeholder: 1, drop_hidden_placeholder: 1})
    batch_error_total += float(batch_error)/num_batches
    #updates ymax and ymin from batch
    if (y_max_ > ymax):
      ymax = y_max_
    if (y_min_ < ymin):
      ymin = y_min_
  avg_test_error = batch_error_total

  
  print("Test Error: ", avg_test_error)
  print("Range of Predicted Outputs: ", ymin," - ", ymax)

  w1_ = W_1.eval()
  b1_ = b_1.eval()
  w2_ = W_2.eval()
  b2_ = b_2.eval()
  w3_ = W_3.eval()
  b3_ = b_3.eval()

 
  #save numpy arrays to CSV
  np.savetxt("two_layer_weights_1.csv", w1_, delimiter=",")
  np.savetxt("two_layer_biases_1.csv", b1_, delimiter=",")
  np.savetxt("two_layer_weights_2.csv", w2_, delimiter=",")
  np.savetxt("two_layer_biases_2.csv", b2_, delimiter=",")
  np.savetxt("two_layer_weights_3.csv", w3_, delimiter=",")
  np.savetxt("two_layer_biases_3.csv", b3_, delimiter=",")


main()