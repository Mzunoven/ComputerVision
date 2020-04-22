import string
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.io
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']

max_iters = 100
# pick a batch size, learning rate
batch_size = 32
learning_rate = 3e-3
hidden_size = 64
##########################
##### your code here #####
##########################

batches = get_random_batches(train_x, train_y, batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
##########################
##### your code here #####
##########################
examples, dimension = train_x.shape
examples, classes = train_y.shape
valid_eg = valid_x.shape[0]

initialize_weights(dimension, hidden_size, params, 'layer1')
initial_W = params['Wlayer1']
initialize_weights(hidden_size, classes, params, 'output')

# fig2 = plt.figure()
# grid = ImageGrid(fig2, 111, nrows_ncols=(8, 8,), axes_pad=0.0)
# for i in range(64):
#     grid[i].imshow(initial_W[:, i].reshape((32, 32)))

train_loss = []
valid_loss = []
train_acc = []
valid_acc = []

# with default settings, you should get loss < 150 and accuracy > 80%
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb, yb in batches:
        # training loop can be exactly the same as q2!
        ##########################
        ##### your code here #####
        ##########################
        h1 = forward(xb, params, 'layer1')
        probs = forward(h1, params, 'output', softmax)
        loss, acc = compute_loss_and_acc(yb, probs)

        total_loss += loss
        total_acc += acc

        delta = probs - yb
        grad = backwards(delta, params, 'output', linear_deriv)
        backwards(grad, params, 'layer1', sigmoid_deriv)

        params['Wlayer1'] -= learning_rate * params['grad_Wlayer1']
        params['blayer1'] -= learning_rate * params['grad_blayer1']
        params['Woutput'] -= learning_rate * params['grad_Woutput']
        params['boutput'] -= learning_rate * params['grad_boutput']

    avg_acc = total_acc / batch_num

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(
            itr, total_loss, avg_acc))

# run on validation set and report accuracy! should be above 75%
# valid_acc: 75.194%

##########################
##### your code here #####
##########################
    valid_h1 = forward(valid_x, params, 'layer1')
    valid_probs = forward(valid_h1, params, 'output', softmax)
    vloss, vacc = compute_loss_and_acc(valid_y, valid_probs)
    train_loss.append(total_loss / examples)
    valid_loss.append(vloss / valid_eg)
    train_acc.append(avg_acc)
    valid_acc.append(vacc)

    print('Validation accuracy: ', vacc)
    if False:  # view the data
        for crop in xb:
            import matplotlib.pyplot as plt
            plt.imshow(crop.reshape(32, 32).T)
            plt.show()

plt.figure(0)
plt.plot(np.arange(max_iters), train_loss, 'r')
plt.plot(np.arange(max_iters), valid_loss, 'b')
plt.legend(['training loss', 'valid loss'])
plt.figure(1)
plt.plot(np.arange(max_iters), train_acc, 'r')
plt.plot(np.arange(max_iters), valid_acc, 'b')
plt.legend(['training accuracy', 'valid accuracy'])
plt.show()
saved_params = {k: v for k, v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3

# visualize weights here
##########################
##### your code here #####
##########################
W_first = params['Wlayer1']
rows, cols = W_first.shape
fig1 = plt.figure()
grid = ImageGrid(fig1, 111, nrows_ncols=(8, 8,), axes_pad=0.0)
for i in range(cols):
    grid[i].imshow(W_first[:, i].reshape((32, 32)))

# Q3.4
confusion_matrix = np.zeros((train_y.shape[1], train_y.shape[1]))

# compute comfusion matrix here
##########################
##### your code here #####
##########################

for i in range(examples):
    xb = train_x[i, :].reshape((1, dimension))
    yb = train_y[i, :].reshape((1, classes))
    h = forward(xb, params, 'layer1')
    probs = forward(h, params, 'output', softmax)
    x_idx = np.argmax(probs[0, :])
    y_idx = np.where(yb == 1)[1][0]
    confusion_matrix[x_idx, y_idx] += 1


# plt.imshow(confusion_matrix, interpolation='nearest')
# plt.grid(True)
# plt.xticks(np.arange(36),
#            string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.yticks(np.arange(36),
#            string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
# plt.show()
