# imports and base params
import itertools
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tensorflow.keras.datasets import mnist
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score



# MNIST dataset parameters.
num_classes = 10 # total classes (0-9 digits).

# Training parameters.
learning_rate = 0.001
batch_size = 128
display_step = 10

# Network parameters.
conv1_filters = 32 # number of filters for 1st conv layer.
conv2_filters = 64 # number of filters for 2nd conv layer.
fc1_units = 256 # number of neurons for 1st fully-connected layer.

# Prepare MNIST data.
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
# Convert to float32.
X_train = tf.cast(tf.constant(X_train), dtype=tf.float32)
X_test = tf.cast(tf.constant(X_test), dtype=tf.float32)

# Normalize images value from [0, 255] to [0, 1].
X_train, X_test = X_train / 255., X_test / 255.

print(X_train.shape)
print(X_test.shape)

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_data = train_data.shuffle(5000).batch(batch_size).prefetch(1)

# test
test_data = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
test_data = test_data.shuffle(5000).batch(batch_size).prefetch(1)

# MODEL
# Create some wrappers for simplicity.
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation.
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2):
    # MaxPool2D wrapper.
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], 
                          strides=[1, k, k, 1], padding='SAME')

# Store layers weight & bias

# A random value generator to initialize weights.
random_normal = tf.initializers.RandomNormal()

weights = {
    # Conv Layer 1: 5x5 conv, 1 input, 32 filters (MNIST has 1 color channel only).
    'wc1': tf.Variable(random_normal([5, 5, 1, conv1_filters])),
    # Conv Layer 2: 5x5 conv, 32 inputs, 64 filters.
    'wc2': tf.Variable(random_normal([5, 5, conv1_filters, conv2_filters])),
    # FC Layer 1: 7*7*64 inputs, 1024 units.
    'wd1': tf.Variable(random_normal([7*7*64, fc1_units])),
    # FC Out Layer: 1024 inputs, 10 units (total number of classes)
    'wout': tf.Variable(random_normal([fc1_units, num_classes]))
}

biases = {
    'bc1': tf.Variable(tf.zeros([conv1_filters])),
    'bc2': tf.Variable(tf.zeros([conv2_filters])),
    'bd1': tf.Variable(tf.zeros([fc1_units])),
    'bout': tf.Variable(tf.zeros([num_classes]))
}

# Create model
def conv_net(x):
    
    # Input shape: [-1, 28, 28, 1]. A batch of 28x28x1 (grayscale) images.
    x = tf.reshape(x, [-1, 28, 28, 1])

    # Convolution Layer. Output shape: [-1, 28, 28, 32].
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    
    # Max Pooling (down-sampling). Output shape: [-1, 14, 14, 32].
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer. Output shape: [-1, 14, 14, 64].
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    
    # Max Pooling (down-sampling). Output shape: [-1, 7, 7, 64].
    conv2 = maxpool2d(conv2, k=2)

    # Reshape conv2 output to fit fully connected layer input, Output shape: [-1, 7*7*64].
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    
    # Fully connected layer, Output shape: [-1, 1024].
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    # Apply ReLU to fc1 output for non-linearity.
    fc1 = tf.nn.relu(fc1)

    # Fully connected layer, Output shape: [-1, 10].
    out = tf.add(tf.matmul(fc1, weights['wout']), biases['bout'])
    # Apply softmax to normalize the logits to a probability distribution.
    return tf.nn.softmax(out)

# METRICS

# Cross-Entropy loss function.
def cross_entropy(y_pred, y_true):
    # Encode label to a one hot vector.
    y_true = tf.one_hot(y_true, depth=num_classes)
    # Clip prediction values to avoid log(0) error.
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    # Compute cross-entropy.
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# ADAM optimizer.
optimizer = tf.optimizers.Adam(learning_rate)

# training function
def run_train(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        pred = conv_net(x)
        loss = cross_entropy(pred, y)
        
    # Variables to update, i.e. trainable variables.
    trainable_variables = list(weights.values()) + list(biases.values())

    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

# testing function
def run_test(test_loader):
  total_acc = 0
  total_loss = 0
  for step, (batch_x, batch_y) in enumerate(test_loader):
    # predict using the model on test data
    pred = conv_net(batch_x)
    # calc acc and loss
    loss = cross_entropy(pred, batch_y)
    acc = accuracy(pred, batch_y)
    total_acc += acc
    # keras way of dealing with loss
    total_loss += loss / len(batch_y)
  # calc total loss and acc
  total_acc = total_acc / (step + 1)
  total_loss = total_loss / (step + 1)
  return total_loss, total_acc

# Run training for the given number of steps.
# one epoch only for demonstration
epochs = 5
epoch_loss = []
epoch_acc = []

for epoch in range(epochs):
  for step, (batch_x, batch_y) in enumerate(train_data):
      # Run the optimization to update W and b values.
      run_train(batch_x, batch_y)
      
      if step % display_step == 0:
          pred = conv_net(batch_x)
          loss = cross_entropy(pred, batch_y)
          acc = accuracy(pred, batch_y)
          print("TRAIN: step: %i, loss: %f, accuracy: %f" % (step + 1, loss / len(batch_y), acc))

  test_loss, test_acc = run_test(test_data)
  print("TEST: epoch: %i, loss: %f, accuracy: %f" % (epoch + 1, test_loss, test_acc))
  # add acc to later be plotted
  epoch_loss.append(test_loss)
  epoch_acc.append(test_acc)

def plot_history(loss, acc):
  fig = plt.figure(figsize=(6, 2))
  ax = fig.gca()
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.plot(acc)
  plt.title("model accuracy")
  plt.ylabel("accuracy")
  plt.xlabel("epoch")
  fig = plt.figure(figsize=(6, 2))
  ax = fig.gca()
  ax.xaxis.set_major_locator(MaxNLocator(integer=True))
  plt.plot(loss)
  plt.title("model loss")       
  plt.ylabel("loss") 
  plt.xlabel("epoch")
  plt.show()

plot_history(epoch_loss, epoch_acc)

# Evaluate the model on test set
def evaluate(test_loader):
  total_acc = 0
  total_loss = 0
  for step, (batch_x, batch_y) in enumerate(test_loader):
    # predict using the model on test data
    pred = conv_net(batch_x)
    # concate the shuffled labels and results
    if step == 0:
      preds = pred
      preds_true = batch_y
    else:
      preds = tf.concat([preds, pred], 0)
      preds_true = tf.concat([preds_true, batch_y], 0)
    # calc acc and loss
    loss = cross_entropy(pred, batch_y)
    acc = accuracy(pred, batch_y)
    total_acc += acc
    total_loss += loss / len(batch_y)
  # calc total loss and acc
  total_acc = total_acc / (step + 1)
  total_loss = total_loss / (step + 1)
  return total_loss, total_acc, preds, preds_true

loss, acc, pred, pred_t = evaluate(test_data)

rep = classification_report(pred_t, np.argmax(pred, axis=1))
acc = accuracy_score(pred_t, np.argmax(pred, axis=1))
print(rep)
print(acc)

# confusion
classes = ['0', '1','2','3', '4', '5', '6', '7', '8', '9']
cm = confusion_matrix(pred_t, np.argmax(pred, axis=1))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Create a dict of variables to save.
vars_to_save = {'optimizer': optimizer}

for key, value in weights.items():
  vars_to_save[key] = value

for key, value in biases.items():
  vars_to_save[key] = value

vars_to_save.keys()

# Save weights and optimizer variables.
# TF Checkpoint, pass the dict as **kwargs.
checkpoint = tf.train.Checkpoint(**vars_to_save)
# TF CheckpointManager to manage saving parameters.
saver = tf.train.CheckpointManager(
      checkpoint, directory="./tf-ckpt", max_to_keep=5)

# Save variables.
saver.save()
