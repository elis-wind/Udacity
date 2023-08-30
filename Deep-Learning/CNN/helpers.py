import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision import transforms as T
import multiprocessing
from pathlib import Path
import os
import math
import random
from tqdm import tqdm


def visualize_filters(filters):
    
    fig, subs = plt.subplots(1, len(filters), figsize=(10, 5))
    
    for i, ax in enumerate(subs.flatten()):    
        
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i+1))
        ax.axis("off")
        
        width, height = filters[i].shape
        
        for x in range(width):
            
            for y in range(height):
                
                ax.annotate(str(filters[i][x][y]), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if filters[i][x][y]<0 else 'black')


def show_feature_maps(input_img, feature_maps, filters):
    
    # Setup visualization grid
    gs_kw = dict(height_ratios=[2, 0.5, 2])
    fig, subs_dict = plt.subplot_mosaic(
        '''
        L....AAAA....
        M.B..C..D..E.
        NFFFGGGHHHIII
        ''',
        figsize=(15, 10),
        gridspec_kw=gs_kw
    )
    
    # plot original image
    subs_dict['A'].imshow(input_img, cmap='gray')
    subs_dict['A'].axis("off")

    # visualize all filters
    for i, p in enumerate(['B', 'C', 'D', 'E']):
        ax = subs_dict[p]
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i+1))
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Visualize feature maps corresponding to each filter
    for i, p in enumerate(['F', 'G', 'H', 'I']):
        ax = subs_dict[p]
        ax.imshow(np.squeeze(feature_maps[0, i].data.numpy()), cmap="gray")
        ax.set_title("Output %s" % str(i + 1))
        ax.axis("off")
    
    subs_dict['L'].text(0, 0.5, "Input image", {'va': 'center'}, rotation=90, fontsize=15)
    subs_dict['L'].axis("off")
    subs_dict['M'].text(0, 0.5, "Conv layer filters", {'va': 'center'},rotation=90, fontsize=15)
    subs_dict['M'].axis("off")
    subs_dict['N'].text(0, 0.5, "Feature maps", {'va': 'center'}, rotation=90, fontsize=15)
    subs_dict['N'].axis("off")
    
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout()

    
def show_feature_maps_full(input_img, model, filters):

    # Setup visualization grid
    gs_kw = dict(height_ratios=[2, 0.5, 2, 2, 2])
    fig, subs_dict = plt.subplot_mosaic(
        """
        1....AAAA....
        2.B..C..D..E.
        3FFFGGGHHHIII
        4LLLMMMNNNOOO
        5PPPQQQRRRSSS
        """,
        figsize=(15, 12),
        gridspec_kw=gs_kw,
    )

    # plot original image
    subs_dict["A"].matshow(input_img, cmap="gray")
    subs_dict["A"].axis("off")

    # visualize all filters
    for i, p in enumerate(["B", "C", "D", "E"]):
        ax = subs_dict[p]
        ax.matshow(filters[i], cmap="gray")
        ax.set_title("Filter %s" % str(i + 1))
        ax.set_xticks([])
        ax.set_yticks([])

    # Visualize feature maps corresponding to each filter
    # Get feature maps (only convolution)
    x = (
        torch.from_numpy(input_img)
        # Add one dimension for batch
        .unsqueeze(0)
        # Add dimension for n_channels
        .unsqueeze(1)
    )
    feature_maps = model.conv_layer(x)

    for i, p in enumerate(["F", "G", "H", "I"]):
        ax = subs_dict[p]
        ax.matshow(np.squeeze(feature_maps[0, i].data.numpy()), cmap="gray")
        ax.set_title("Output %s" % str(i + 1))
        ax.axis("off")

    # Visualize feature maps after activation
    feature_maps = model.activation(model.conv_layer(x))
    for i, p in enumerate(["L", "M", "N", "O"]):
        ax = subs_dict[p]
        ax.matshow(np.squeeze(feature_maps[0, i].data.numpy()), cmap="gray")
        ax.set_title("Output %s" % str(i + 1))
        ax.axis("off")

    # Visualize feature maps after max pool
    final_res = model(x)
    for i, p in enumerate(["P", "Q", "R", "S"]):

        data = np.squeeze(final_res[0, i].data.numpy())

        ax = subs_dict[p]
        ax.matshow(data, cmap="gray")
        ax.set_title("Output %s" % str(i + 1))
        ax.axis("off")

        half_x_diff = (feature_maps[0, i].data.shape[1] - data.shape[1]) / 2
        half_y_diff = (feature_maps[0, i].data.shape[0] - data.shape[0]) / 2
        ax.set_xlim((lim - half_x_diff for lim in subs_dict["L"].get_xlim()))
        ax.set_ylim((lim - half_y_diff for lim in subs_dict["L"].get_ylim()))

    subs_dict["1"].text(
        0, 0.5, "Input image", {"va": "center"}, rotation=90, fontsize=15
    )
    subs_dict["1"].axis("off")
    subs_dict["2"].text(
        0, 0.5, "Conv layer filters", {"va": "center"}, rotation=90, fontsize=15
    )
    subs_dict["2"].axis("off")
    subs_dict["3"].text(
        0, 0.5, "Feature maps\nafter conv", {"va": "center"}, rotation=90, fontsize=15
    )
    subs_dict["3"].axis("off")
    subs_dict["4"].text(
        0, 0.5, "Feature maps\nafter ReLU", {"va": "center"}, rotation=90, fontsize=15
    )
    subs_dict["4"].axis("off")
    subs_dict["5"].text(
        0,
        0.5,
        "Feature maps\nafter MaxPool",
        {"va": "center"},
        rotation=90,
        fontsize=15,
    )
    subs_dict["5"].axis("off")

    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.tight_layout()


def get_train_val_data_loaders(batch_size, valid_size, transforms, num_workers):

    # Get the CIFAR10 training dataset from torchvision.datasets and set the transforms
    # We will split this further into train and validation in this function
    train_data = datasets.CIFAR10("data", train=True, download=True, transform=transforms)

    # Compute how many items we will reserve for the validation set
    n_tot = len(train_data)
    split = int(np.floor(valid_size * n_tot))

    # compute the indices for the training set and for the validation set
    shuffled_indices = torch.randperm(n_tot)
    train_idx, valid_idx = shuffled_indices[split:], shuffled_indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    # prepare data loaders (combine dataset and sampler)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers
    )

    return train_loader, valid_loader


def get_test_data_loader(batch_size, transforms, num_workers):
    # We use the entire test dataset in the test dataloader
    test_data = datasets.CIFAR10("data", train=False, download=True, transform=transforms)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers
    )

    return test_loader


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one epoch of training
    """

    # Move model to GPU if available
    if torch.cuda.is_available():
        model.cuda()  # -

    # Set the model in training mode
    # (so all layers that behave differently between training and evaluation,
    # like batchnorm and dropout, will select their training behavior)
    model.train()  # -

    # Loop over the training data
    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU if available
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad()  # -
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)  # =
        # 3. calculate the loss
        loss_value = loss(output, target)  # =
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward()  # -
        # 5. perform a single optimization step (parameter update)
        optimizer.step()  # -

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """

    # During validation we don't need to accumulate gradients
    with torch.no_grad():

        # set the model to evaluation mode
        # (so all layers that behave differently between training and evaluation,
        # like batchnorm and dropout, will select their evaluation behavior)
        model.eval()  # -

        # If the GPU is available, move the model to the GPU
        if torch.cuda.is_available():
            model.cuda()

        # Loop over the validation dataset and accumulate the loss
        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            # move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)  # =
            # 2. calculate the loss
            loss_value = loss(output, target)  # =

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses()
    else:
        liveloss = None

    # Loop over the epochs and keep track of the minimum of the validation loss
    valid_loss_min = None
    logs = {}

    for epoch in range(1, n_epochs + 1):

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path)  # -

            valid_loss_min = valid_loss

        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss

            liveloss.update(logs)
            liveloss.send()

            
def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    # we do not need the gradients
    with torch.no_grad():

        # set the model to evaluation mode
        model.eval()  # -

        # if the GPU is available, move the model to the GPU
        if torch.cuda.is_available():
            model = model.cuda()

        # Loop over test dataset
        # We also accumulate predictions and targets so we can return them
        preds = []
        actuals = []
        
        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU if available
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data)  # =
            # 2. calculate the loss
            loss_value = loss(logits, target).detach()  # =

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # NOTE: the predicted class is the index of the max of the logits
            pred = logits.data.max(1, keepdim=True)[1]  # =

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)
            
            preds.extend(pred.data.cpu().numpy().squeeze())
            actuals.extend(target.data.view_as(pred).cpu().numpy().squeeze())

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss, preds, actuals


def plot_confusion_matrix(pred, truth, classes):

    gt = pd.Series(truth, name='Ground Truth')
    predicted = pd.Series(pred, name='Predicted')

    confusion_matrix = pd.crosstab(gt, predicted)
    confusion_matrix.index = classes
    confusion_matrix.columns = classes
    
    fig, sub = plt.subplots()
    with sns.plotting_context("notebook"):

        ax = sns.heatmap(
            confusion_matrix, 
            annot=True, 
            fmt='d',
            ax=sub, 
            linewidths=0.5, 
            linecolor='lightgray', 
            cbar=False
        )
        ax.set_xlabel("truth")
        ax.set_ylabel("pred")

    

    return confusion_matrix


def get_data_loaders(batch_size, val_fraction=0.2):
    
    transform = transforms.ToTensor()
    
    num_workers = multiprocessing.cpu_count()
    
    # Get train, validation and test
    
    # Let's start with train and validation
    trainval_data = datasets.MNIST(
        root="data", train=True, download=True, transform=transform
    )

    # Split in train and validation
    # NOTE: we set the generator with a fixed random seed for reproducibility
    train_len = int(len(trainval_data) * (1 - val_fraction))
    val_len = len(trainval_data) - train_len
    print(f"Using {train_len} examples for training and {val_len} for validation")
    train_subset, val_subset = torch.utils.data.random_split(
        trainval_data, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_subset, shuffle=True, batch_size=batch_size, num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_subset, shuffle=False, batch_size=batch_size, num_workers=num_workers
    )

    # Get test data
    test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, num_workers=num_workers
    )
    print(f"Using {len(test_data)} for testing")
    
    return {
        'train': train_loader,
        'valid': val_loader,
        'test': test_loader
    }


def seed_all(seed=42):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def anomaly_detection_display(df):
    
    df.sort_values(by='loss', ascending=False, inplace=True)
    
    fig, sub = plt.subplots()
    df['loss'].hist(bins=100)
    sub.set_yscale('log')
    sub.set_xlabel("Score (loss)")
    sub.set_ylabel("Counts per bin")
    fig.suptitle("Distribution of score (loss)")
    
    fig, subs = plt.subplots(2, 20, figsize=(20, 3))

    for img, sub in zip(df['image'].iloc[:20], subs[0].flatten()):
        sub.imshow(img[0, ...], cmap='gray')
        sub.axis("off")

    for rec, sub in zip(df['reconstructed'].iloc[:20], subs[1].flatten()):
        sub.imshow(rec[0, ...], cmap='gray')
        sub.axis("off")

    fig.suptitle("Most difficult to reconstruct")
    subs[0][0].axis("on")
    subs[0][0].set_xticks([])
    subs[0][0].set_yticks([])
    subs[0][0].set_ylabel("Input")

    subs[1][0].axis("on")
    subs[1][0].set_xticks([])
    subs[1][0].set_yticks([])
    _ = subs[1][0].set_ylabel("Reconst")
    
    fig, subs = plt.subplots(2, 20, figsize=(20, 3))

    sample = df.iloc[7000:].sample(20)

    for img, sub in zip(sample['image'], subs[0].flatten()):
        sub.imshow(img[0, ...], cmap='gray')
        sub.axis("off")

    for rec, sub in zip(sample['reconstructed'], subs[1].flatten()):
        sub.imshow(rec[0, ...], cmap='gray')
        sub.axis("off")

    fig.suptitle("Sample of in-distribution numbers")
    subs[0][0].axis("on")
    subs[0][0].set_xticks([])
    subs[0][0].set_yticks([])
    subs[0][0].set_ylabel("Input")

    subs[1][0].axis("on")
    subs[1][0].set_xticks([])
    subs[1][0].set_yticks([])
    _ = subs[1][0].set_ylabel("Reconst")


seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def get_data_location():
    """
    Find the location of the dataset, either locally or in the Udacity workspace
    """

    if os.path.exists("flowers"):
        data_folder = "flowers"
    elif os.path.exists("/data/DLND/C2/flowers"):
        data_folder = "/data/DLND/C2/flowers"
    else:
        raise IOError("Please download the dataset first")

    return data_folder


# Compute image normalization
def compute_mean_and_std():
    """
    Compute per-channel mean and std of the dataset (to be used in transforms.Normalize())
    """

    cache_file = "mean_and_std.pt"
    if os.path.exists(cache_file):
        print(f"Reusing cached mean and std")
        d = torch.load(cache_file)

        return d["mean"], d["std"]

    folder = get_data_location()
    ds = datasets.ImageFolder(
        folder, transform=T.Compose([T.ToTensor()])
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=1, num_workers=multiprocessing.cpu_count()
    )

    mean = 0.0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing mean", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dl.dataset)

    var = 0.0
    npix = 0
    for images, _ in tqdm(dl, total=len(ds), desc="Computing std", ncols=80):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1)) ** 2).sum([0, 2])
        npix += images.nelement()

    std = torch.sqrt(var / (npix / 3))

    # Cache results so we don't need to redo the computation
    torch.save({"mean": mean, "std": std}, cache_file)

    return mean, std


def get_transforms(rand_augment_magnitude):

    # These are the per-channel mean and std of CIFAR-10 over the dataset
    mean, std = compute_mean_and_std()

    # Define our transformations
    return {
        "train": T.Compose(
            [
                # All images in CIFAR-10 are 32x32. We enlarge them a bit so we can then
                # take a random crop
                T.Resize(256),
                
                # take a random part of the image
                T.RandomCrop(224),
                
                # Horizontal flip is not part of RandAugment according to the RandAugment
                # paper
                T.RandomHorizontalFlip(0.5),
                
                # RandAugment has 2 main parameters: how many transformations should be
                # applied to each image, and the strength of these transformations. This
                # latter parameter should be tuned through experiments: the higher the more
                # the regularization effect
                T.RandAugment(
                    num_ops=2,
                    magnitude=rand_augment_magnitude,
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        ),
        "valid": T.Compose(
            [
                # Both of these are useless, but we keep them because
                # in a non-academic dataset you will need them
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean, std),
            ]
        )
    }


def get_data_loaders2(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1, rand_augment_magnitude: int = 9
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    if num_workers == -1:
        # Use all cores
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # YOUR CODE HERE:
    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # HINT: resize the image to 256 first, then crop them to 224, then add the
    # appropriate transforms for that step
    data_transforms = get_transforms(rand_augment_magnitude)
#     data_transforms = {
#         "train": transforms.Compose(
#             # YOUR CODE HERE
#             [  # -
#                 transforms.Resize(256),  # -
#                 transforms.RandomCrop(224, padding_mode="reflect", pad_if_needed=True),  # -
#                 transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),  # -
#                 transforms.ToTensor(),  # -
#                 transforms.Normalize(mean, std),  # -
#             ]  # -
#         ),
#         "valid": transforms.Compose(
#             # YOUR CODE HERE
#             [  # -
#                 transforms.Resize(256),  # -
#                 transforms.CenterCrop(224),  # -
#                 transforms.ToTensor(),  # -
#                 transforms.Normalize(mean, std),  # -
#             ]  # -
#         )
#     }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path,
        # YOUR CODE HERE: add the appropriate transform that you defined in
        # the data_transforms dictionary
        transform=data_transforms["train"]  # -
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path,
        # YOUR CODE HERE: add the appropriate transform that you defined in
        # the data_transforms dictionary
        transform=data_transforms["valid"]  # -
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.SubsetRandomSampler(valid_idx)  # =

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        # YOUR CODE HERE
        valid_data,  # -
        batch_size=batch_size,  # -
        sampler=valid_sampler,  # -
        num_workers=num_workers,  # -
    )

    return data_loaders