import torch
# from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt

# 后续就没用到
# class Normal(object):
#     def __init__(self, mu, sigma, log_sigma, v=None, r=None):
#         self.mu = mu
#         self.sigma = sigma  # either stdev diagonal itself, or stdev diagonal from decomposition
#         self.logsigma = log_sigma
#         dim = mu.get_shape()
#         if v is None:
#             v = torch.FloatTensor(*dim)
#         if r is None:
#             r = torch.FloatTensor(*dim)
#         self.v = v
#         self.r = r


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.relu(self.linear2(x))


class VAE(torch.nn.Module):
    latent_dim = 8  # 隐变量维度

    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        # 编码器到均值和log方差的两个网络, 且默认知道编码器的最后一层维度为100, 隐变量维度为8
        self._enc_mu = torch.nn.Linear(100, 8)
        self._enc_log_sigma = torch.nn.Linear(100, 8)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = mu
        self.z_sigma = sigma

        return mu + sigma * std_z  # Reparameterization trick
        # return mu + sigma * Variable(std_z, requires_grad=False)  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)  # 编码器的输出
        z = self._sample_latent(h_enc)
        return self.decoder(z)


# 计算KL散度误差项
def latent_loss(z_mean, z_stddev):
    mean_sq = z_mean * z_mean
    stddev_sq = z_stddev * z_stddev
    return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)


if __name__ == '__main__':

    input_dim = 28 * 28
    batch_size = 32

    transform = transforms.Compose(
        [transforms.ToTensor()])
    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))

    encoder = Encoder(input_dim, 100, 100)  # 隐藏层维度为100, 输出维度为100
    decoder = Decoder(8, 100, input_dim)  # 隐藏层维度为100
    vae = VAE(encoder, decoder)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in range(100):
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data
            inputs.resize_(batch_size, input_dim)
            optimizer.zero_grad()
            dec = vae(inputs)  # 获得的输出端解码
            ll = latent_loss(vae.z_mean, vae.z_sigma)
            loss = criterion(dec, inputs) + ll  # 重建误差 + KL散度误差
            loss.backward()
            optimizer.step()
            l = loss.item()
            # l = loss.data[0]
            """
            考虑在 PyTorch0.4.0 版本之前广泛使用的 total_loss + = loss.data [0] 模式
            Loss 是一个包含张量(1, )的Variable, 但是在新发布的 0.4.0 版本中. loss 是一个0维标量, 对于标量的索引是没有意义的
            使用 loss.item（）从标量中获取 Python 数字
            """
        print(epoch, l)

    plt.imshow(vae(inputs).data[0].numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)
