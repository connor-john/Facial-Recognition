# importing the libraries
import numpy as np
import torch

from model import *
from preprocessing import prepare_data
from datetime import datetime

# Hyper params
H = 60
W = 80
batch_size = 64
datapath = ''
feature_dim = 50

# Generator training/testing loop
# each batch send 1 pair of each subject and same number non-matching
def run_generator(positives, negatives, images, train = True):

    n_batches = int(np.ceil(len(positives) / batch_size))

    while True:
        # Shuffle positives if training
        if train:
            np.random.shuffle(positives)
        
        n_samples = batch_size * 2
        shape = (n_samples, H, W)
        # x: creating inputs
        # y: creating targets
        x_batch_1 = np.zeros(shape)
        x_batch_2 = np.zeros(shape)
        y_batch = np.zeros(n_samples)

        for i in range(n_batches):

            pos_batch_indices = positives[i * batch_size: (i + 1) * batch_size]

            # fill x, y
            j = 0
            for idx1, idx2 in pos_batch_indices:
                x_batch_1[j] = images[idx1]
                x_batch_2[j] = images[idx2]
                y_batch[j] = 1 # match
                j += 1
            
            # negative samples
            neg_indices = np.random.choice(len(negatives), size=len(pos_batch_indices), replace=False)
            for neg in neg_indices:
                idx1, idx2 = negatives[neg]
                x_batch_1[j] = images[idx1]
                x_batch_2[j] = images[idx2]
                y_batch[j] = 0 # non-match
                j += 1
            
            x1 = x_batch_1[:j]
            x2 = x_batch_2[:j]
            y = y_batch[:j]

            # reshape
            x1 = x1.reshape(-1, 1, H, W)
            x2 = x2.reshape(-1, 1, H, W)

            # convert to torch tensor
            x1 = torch.from_numpy(x1).float()
            x2 = torch.from_numpy(x2).float()
            y = torch.from_numpy(y).float()

            yield [x1, x2], y

# Training loop
# Batch gradient descent
def train(model, criterion, optimizer, train_gen, test_gen, train_steps_per_epoch, test_steps_per_epoch, epochs):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for i in range(epochs):

        t0 = datetime.now()
        train_loss = []
        steps = 0
        for (x1, x2), targets in train_gen:

            # data to GPU
            x1, x2, targets = x1.to(device), x2.to(device), targets.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
                
            # Backward and optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

            # Update steps
            steps += 1
            if steps >= train_steps_per_epoch:
                break

        # Get train loss and test loss
        train_loss = np.mean(train_loss)
        
        test_loss = []
        steps = 0
        for (x1, x2), targets in test_gen:

            x1, x2, targets = x1.to(device), x2.to(device), targets.to(device)
            outputs = model(x1, x2)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
            steps += 1
            if steps >= train_steps_per_epoch:
                break

        test_loss = np.mean(test_loss)

        # Save losses
        train_losses[i] = train_loss
        test_losses[i] = test_loss
        
        dt = datetime.now() - t0
        print(f'epoch {i+1}/{epochs} | train_loss: {train_loss:.4f} | test_oss: {test_loss:.4f} | duration: {dt}')
    
    return train_losses, test_losses

# main
if __name__ == '__main__':

    # initialise
    train_images, train_labels, test_images, test_labels, train_positives, train_negatives, test_positives, test_negatives = prepare_data(datapath, H, W)
    model = SiameseModel(feature_dim)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())

    train_steps = int(np.ceil(len(train_positives) / batch_size))
    test_steps = int(np.ceil(len(test_positives) / batch_size))

    # training loop
    train_losses, test_losses = train(model, contrastive_loss, optimizer, 
                                        run_generator(train_positives, train_negatives, train_images), 
                                        run_generator(test_positives, test_negatives, test_images, train = False), 
                                        train_steps, test_steps, epochs=20)
