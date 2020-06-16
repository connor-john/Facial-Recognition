# importing the libraries
import numpy as np
import torch

# Hyper params
H = 60
W = 80

# Generator training/testing loop
# each batch send 1 pair of each subject and same number non-matching
def run_generator(positives, negatives, images, batch_size = 64, train = True):

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
