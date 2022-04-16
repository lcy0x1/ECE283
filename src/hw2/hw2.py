import torch  # root package
from torch.utils.data import Dataset, DataLoader  # dataset representation and loading
import torch.autograd as autograd  # computation graph
from torch import Tensor  # tensor node in the computation graph
import torch.nn as nn  # neural networks
import torch.nn.functional as F  # layers, activations and more
import torch.optim as optim  # optimizers e.g. gradient descent, ADAM, etc.
from torch.jit import script, trace  # hybrid frontend decorator and tracing jit
import matplotlib.pyplot as plt

import math
import numpy as np

from gaussian_mixture import GaussianMixture


class Net(nn.Module):

    def __init__(self, n_in, n_hidden):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, 1)

    def forward(self, x):
        x = self.fc1.forward(x)
        x = F.relu(x)
        x = self.fc2.forward(x)
        x = torch.sigmoid(x)
        return x


def a_from_eigen_angle(eig_1, eig_2, theta):
    eig_vector = np.transpose(np.array([[math.cos(theta), math.sin(theta)],
                                        [-math.sin(theta), math.cos(theta)]]))
    eig_value = np.diag([math.sqrt(eig_1), math.sqrt(eig_2)])
    return np.matmul(eig_vector, eig_value)


if __name__ == '__main__':
    mixture_0 = GaussianMixture(2)
    mixture_0.add_gaussian(2 / 3, np.array([4, 0]), a_from_eigen_angle(1, 4, 0))
    mixture_0.add_gaussian(1 / 3, np.array([-3, 3]), a_from_eigen_angle(1, 4, math.pi / 4))

    mixture_1 = GaussianMixture(2)
    mixture_1.add_gaussian(3 / 4, np.array([0, 0]), a_from_eigen_angle(1, 2, math.pi / 3))
    mixture_1.add_gaussian(1 / 4, np.array([-6, -4]), a_from_eigen_angle(2, 1, math.pi / 4))

    sample_0 = mixture_0.get_random_data(200, 0)
    sample_1 = mixture_1.get_random_data(200, 0)


    def gen_sample(data_0, data_1, n):
        return torch.from_numpy(np.concatenate((data_0[0:n, :], data_1[0:n, :])))

    n_sample = 200
    l2_lambda = 1e-2
    lrate = 1e-5

    net = Net(2, 32)
    optimizer = optim.SGD(net.parameters(), lr=lrate)
    optimizer.zero_grad()
    criterion = nn.CrossEntropyLoss()
    data = gen_sample(sample_0, sample_1, n_sample)
    label = torch.from_numpy(np.concatenate((np.zeros((n_sample,), np.single), np.ones((n_sample,), np.single)))).view(
        [400, 1])

    prev = 1e3
    lv = 1
    step = 0
    list_lv = []
    list_ac0 = []
    list_ac1 = []
    while abs(prev - lv) > lv * 1e-9:
        prev = lv
        output = net.forward(data)
        loss = criterion(output, label)
        l2_norm = sum(p.pow(2).sum() for p in net.parameters())
        loss = loss + l2_lambda * l2_norm
        lv = loss.item()
        ac_0 = int(torch.count_nonzero(output.data[0:n_sample] > 0.5)) / n_sample
        ac_1 = int(torch.count_nonzero(output.data[n_sample:-1] < 0.5)) / n_sample
        loss.backward()
        optimizer.step()
        step += 1
        list_lv.append(lv)
        list_ac0.append(ac_0)
        list_ac1.append(ac_1)

    plt.plot(list_lv)
    plt.plot(list_ac0)
    plt.plot(list_ac1)
    plt.ylim([0, 1])

