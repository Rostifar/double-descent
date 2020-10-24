import torchvision
import pickle
import os
import sys
import copy

import numpy as np
from torchvision.datasets import cifar


class CIFAR10Noisy(cifar.CIFAR10):
    # hack
    def _check_integrity(self):
        return True

    def __init__(self, root, noise=None, train=True, transform=None, target_transform=None):
        super(cifar.CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        #if download:
        # TODO: remove this
        #self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        c_1 = 0
        c_2 = 0

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')

                c = copy.deepcopy(entry)
                self.data.append(entry['data'])
                if 'labels' in entry:
                    if noise is not None:
                        for j in range(len(entry['labels'])):
                            switch = np.random.binomial(1, p=noise)
                            if switch:
                                label = int(entry['labels'][j])
                                new_label = np.random.choice(10, 1, p=[(1/9) if i != label else 0.0 for i in range(10)])[0]
                                assert new_label != label
                                entry['labels'][j] = new_label
                                c_1 += 1
                            c_2 += 1
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

            #print(float(c_1)/float(c_2))
            #if noise is not None:
            #    with open(file_path, 'wb') as f:
            #        pickle.dump(entry, f)

            #with open(file_path, 'rb') as f:
            #entry = pickle.load(f, encoding='latin1')
            #assert(entry['labels'] != c['labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()


class CIFAR100Noisy(CIFAR10Noisy):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """
    base_folder = 'cifar-100-python'
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = 'eb9058c3a382ffc7106e4002c42a8d85'
    train_list = [
        ['train', '16019d7e3df5f24257cddd939b257f8d'],
    ]

    test_list = [
        ['test', 'f0ef6b0ae62326f3e7ffdfab6717acfc'],
    ]
    meta = {
        'filename': 'meta',
        'key': 'fine_label_names',
        'md5': '7973b15100ade9c7d40fb424638fde48',
    }


