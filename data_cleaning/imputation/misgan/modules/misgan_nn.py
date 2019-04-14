import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# TODO: this should return the masked matrix with all missing value replaced by tau
def mask_data(data, mask, tau=0):
    return mask * data + (1 - mask) * tau

# Must sub-class ConvGenerator to provide transform()
class ConvGenerator(nn.Module):
    def __init__(self, K, D, latent_size=128):
        super().__init__()

        self.K = K
        self.D = D
        self.DIM = 64
        self.latent_size = latent_size

        self.preprocess = nn.Sequential(
            nn.Linear(latent_size, 4 * 4 * 4 * self.DIM),
            nn.ReLU(True),
        )
        self.block1 = nn.Sequential(
            nn.ConvTranspose2d(4 * self.DIM, 2 * self.DIM, 5),
            nn.ReLU(True),
        )
        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(2 * self.DIM, self.DIM, 5),
            nn.ReLU(True),
        )
        self.deconv_out = nn.ConvTranspose2d(self.DIM, 1, 6, stride=1)

    def forward(self, input):
        net = self.preprocess(input)
        net = net.view(-1, 4 * self.DIM, 4, 4)
        net = self.block1(net)
        net = net[:, :, :7, :7]
        net = self.block2(net)
        net = self.deconv_out(net)
        return self.transform(net).view(-1, 1, self.K, self.D)

#10,64,11,11 -> 10, 1, 28, 28
class ConvDataGenerator(ConvGenerator):
    def __init__(self, K, D, latent_size=128):
        super().__init__(K, D, latent_size=latent_size)
        self.K = K
        self.D = D
        self.transform = lambda x: torch.sigmoid(x)


class ConvMaskGenerator(ConvGenerator):
    def __init__(self, K, D, latent_size=128, temperature=.66):
        super().__init__(K, D, latent_size=latent_size)
        self.K = K
        self.D = D
        self.transform = lambda x: torch.sigmoid(x / temperature)

class ConvCritic(nn.Module):
    def __init__(self, K, D):
        super().__init__()

        self.K = K
        self.D = D
        self.DIM = 64
        main = nn.Sequential(
            nn.Conv2d(1, self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(self.DIM, 2 * self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
            nn.Conv2d(2 * self.DIM, 4 * self.DIM, 5, stride=2, padding=2),
            nn.ReLU(True),
        )
        self.main = main
        self.output = nn.Linear(4 * 4 * 10 * self.DIM, 1)

    def forward(self, input):
        input = input.view(-1, 1, self.K, self.D)
        net = self.main(input)
        net = net.view(-1, 4 * 4 * 10 * self.DIM)
        net = self.output(net)
        return net.view(-1)

class CriticUpdater:
    def __init__(self, critic, critic_optimizer, device, batch_size=10, gp_lambda=10):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.gp_lambda = gp_lambda
        # Interpolation coefficient
        self.eps = torch.empty(batch_size, 1, 1, 1, device=device)
        # For computing the gradient penalty
        self.ones = torch.ones(batch_size).to(device)

    def __call__(self, real, fake):
        real = real.detach()
        fake = fake.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)
        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()
        grad_d = grad(self.critic(interp), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        w_dist = self.critic(fake).mean() - self.critic(real).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()


class Imputer(nn.Module):
    def __init__(self, K, D, arch=(512, 512)):
        super().__init__()
        self.K = K
        self.D = D
        self.fc = nn.Sequential(
            nn.Linear(self.K * self.D, arch[0]),
            nn.ReLU(),
            nn.Linear(arch[0], arch[1]),
            nn.ReLU(),
            nn.Linear(arch[1], arch[0]),
            nn.ReLU(),
            nn.Linear(arch[0], self.K * self.D),
        )

    def forward(self, data, mask, noise):
        net = data * mask + noise * (1 - mask)
        net = net.view(data.shape[0], -1)
        net = self.fc(net)
        net = torch.sigmoid(net).view(data.shape)
        return data * mask + net * (1 - mask)

    def inference(self, data, mask):
        net = data * mask
        net = net.view(data.shape[0], -1)
        net = self.fc(net)
        net = torch.sigmoid(net).view(data.shape)
        return data * mask + net * (1 - mask)
