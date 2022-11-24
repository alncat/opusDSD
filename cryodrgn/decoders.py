import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from . import utils

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):

        super(VectorQuantizer, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        input_shape = inputs.shape #(B, C)

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost

        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()

        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        #inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)

            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        loss = self._commitment_cost * e_latent_loss

        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from BHWC -> BCHW
        return loss, quantized, perplexity, encodings

class ConvTemplate(nn.Module):
    def __init__(self, in_dim=256, outchannels=1, templateres=128):

        super(ConvTemplate, self).__init__()

        self.zdim = in_dim
        self.outchannels = outchannels
        self.templateres = templateres

        self.template1 = nn.Sequential(nn.Linear(self.zdim, 512), nn.LeakyReLU(0.2),
                                       nn.Linear(512, 2048), nn.LeakyReLU(0.2))
        template2 = []
        inchannels, outchannels = 2048, 1024
        template2.append(nn.ConvTranspose3d(inchannels, outchannels, 2, 2, 0))
        template2.append(nn.LeakyReLU(0.2))
        inchannels, outchannels = 1024, 512
        template2.append(nn.ConvTranspose3d(inchannels, outchannels, 2, 2, 0))
        template2.append(nn.LeakyReLU(0.2))
        inchannels, outchannels = 512, 256
        for i in range(int(np.log2(self.templateres)) - 3):
            template2.append(nn.ConvTranspose3d(inchannels, outchannels, 4, 2, 1))
            #template2.append(nn.Conv3d(inchannels, outchannels, 3, 1, 1))
            #template2.append(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True))
            template2.append(nn.LeakyReLU(0.2))
            #if inchannels == outchannels:
            inchannels = outchannels
            outchannels = max(inchannels // 2, 16)
            #else:
            #inchannels = outchannels
        template2.append(nn.ConvTranspose3d(inchannels, 1, 4, 2, 1))
        self.template2 = nn.Sequential(*template2)
        for m in [self.template1, self.template2]:
            utils.initseq(m)

    def forward(self, encoding):
        #modules = [module for k, module in self.template2._modules.items()]
        #return checkpoint_sequential(modules, 2, self.template1(encoding).view(-1, 1024, 1, 1, 1))
        return self.template2(self.template1(encoding).view(-1, 2048, 1, 1, 1))

