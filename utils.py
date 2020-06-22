import numpy as np
import matplotlib.pyplot as plt

# calculate accuracy before training
# since the dataset is imbalanced, we'll report tp, tn, fp, fn
def test_accuracy(H, W, test_images, test_positives, test_negatives, predict, threshold=0.85):

    positive_distances = []
    negative_distances = []

    # true postive
    # sensitivity
    tp = 0
    # true negative
    # specificity
    tn = 0
    fp = 0
    fn = 0

    batch_size = 64
    x_batch_1 = np.zeros((batch_size, 1, H, W))
    x_batch_2 = np.zeros((batch_size, 1, H, W))
    n_batches = int(np.ceil(len(test_positives) / batch_size))

    for i in range(n_batches):

        print(f"pos_batch: {i+1}/{n_batches}")
        pos_batch_indices = test_positives[i * batch_size: (i + 1) * batch_size]

        # fill up x_batch and y_batch
        j = 0
        for idx1, idx2 in pos_batch_indices:
            x_batch_1[j,0] = test_images[idx1]
            x_batch_2[j,0] = test_images[idx2]
            j += 1

        x1 = x_batch_1[:j]
        x2 = x_batch_2[:j]
        distances = predict(x1, x2)
        positive_distances += distances.tolist()

        # update tp, tn, fp, fn
        tp += (distances < threshold).sum()
        fn += (distances > threshold).sum()

    n_batches = int(np.ceil(len(test_negatives) / batch_size))

    for i in range(n_batches):

        print(f"neg_batch: {i+1}/{n_batches}")

        neg_batch_indices = test_negatives[i * batch_size: (i + 1) * batch_size]

        # fill up x_batch and y_batch
        j = 0
        for idx1, idx2 in neg_batch_indices:
            x_batch_1[j] = test_images[idx1]
            x_batch_2[j] = test_images[idx2]
            j += 1

        x1 = x_batch_1[:j]
        x2 = x_batch_2[:j]
        distances = predict(x1, x2)
        negative_distances += distances.tolist()

        # update tp, tn, fp, fn
        fp += (distances < threshold).sum()
        tn += (distances > threshold).sum()


    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    print(f"sensitivity: {tpr}, specificity: {tnr}")

    plt.hist(negative_distances, bins=20, density=True, label='negative_distances')
    plt.hist(positive_distances, bins=20, density=True, label='positive_distances')
    plt.axvline(x=threshold, color = 'r', linestyle = 'dashed', linewidth = 3)
    plt.legend()
    plt.show()