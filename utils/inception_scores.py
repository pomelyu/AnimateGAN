# Modified from https://github.com/sbarratt/inception-score-pytorch
import numpy as np
from scipy.stats import entropy

from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models.inception import inception_v3

class ListDataset(Dataset):
    def __init__(self, data):
        assert isinstance(data, list)
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs
    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    if not isinstance(imgs, Dataset):
        imgs = ListDataset(imgs)

    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    def get_pred(x):
        if resize:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
        x = inception_model(x)
        return F.softmax(x, dim=1).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in tqdm(enumerate(dataloader, 0), total=len(dataloader), ascii=True):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

if __name__ == '__main__':
    import torchvision.datasets as dset
    import torchvision.transforms as T

    cifar = dset.CIFAR10(root='data/', download=True,
                         transform=T.Compose([
                             T.Scale(32),
                             T.ToTensor(),
                             T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                         ]))

    cifar_dataset = ListDataset([img[0] for img in cifar])
    print("Calculating Inception Score...")
    res_mean, res_std = inception_score(cifar_dataset, cuda=torch.cuda.is_available(), batch_size=32, resize=True, splits=10)
    print("Inception score mean: {:.3f}, std: {:.3f}".format(res_mean, res_std))
