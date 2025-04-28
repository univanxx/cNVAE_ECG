### SOURCE: https://github.com/NVlabs/NVAE/blob/master/distributions.py


import torch
import torch.nn.functional as F
import numpy as np

from utils import one_hot

 
@torch.jit.script
def soft_clamp5(x: torch.Tensor):
    return x.div(5.).tanh_().mul(5.)    #  5. * torch.tanh(x / 5.) <--> soft differentiable clamp between [-5, 5]


@torch.jit.script 
def sample_normal_jit(mu, sigma):
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps


class Normal:
    def __init__(self, mu, log_sigma, temp=1.):
        self.mu = soft_clamp5(mu)
        log_sigma = soft_clamp5(log_sigma)
        self.sigma = torch.exp(log_sigma) + 1e-2      # we don't need this after soft clamp
        if temp != 1.:
            self.sigma *= temp

    def sample(self):
        return sample_normal_jit(self.mu, self.sigma)

    def sample_given_eps(self, eps):
        return eps * self.sigma + self.mu
    def log_p(self, samples):
        normalized_samples = (samples - self.mu) / (self.sigma)
        log_p = - 0.5 * normalized_samples * normalized_samples - 0.5 * np.log(2 * np.pi)# - torch.log(self.sigma)
        return log_p

    def kl(self, normal_dist):
        term1 = (self.mu - normal_dist.mu) / normal_dist.sigma
        term2 = self.sigma / normal_dist.sigma

        return 0.5 * (term1 * term1 + term2 * term2) - 0.5 - torch.log(term2)


class DiscMixEightLogistic1D:
    def __init__(self, param, num_mix=10, num_bits=8, focal=False):
        B, C, W = param.size()
        self.num_mix = num_mix
        self.logit_probs = param[:, :num_mix, :]                                   # B, M, W
        l = param[:, num_mix:, :].view(B, 32, num_mix, W)                    # B, 32, M, W
        self.means = l[:, :8, :, :]                                          # B, 8, M, W
        self.log_scales = torch.clamp(l[:, 8:16, :, :], min=-7.0)   # B, 8, M, W
        self.coeffs = torch.tanh(l[:, 16:, :, :])              # B, 16, M, W
        self.max_val = 2. ** num_bits - 1
        self.focal = focal
        
    def log_prob(self, samples):
        assert torch.max(samples) <= 1.0 and torch.min(samples) >= 0.0
        # convert samples to be in [-1, 1]
        samples = 2 * samples - 1.0
        B, C, W = samples.size()
        assert C == 8, 'only 8-channel ECGs are considered.'

        samples = samples.unsqueeze(3)                                                  # B, 8*M, W
        samples = samples.expand(-1, -1, -1, self.num_mix).permute(0, 1, 3, 2)   # B, 8, M, W
        
        mean1 = self.means[:, 0, :, :]                                               # B, M, W
        mean2 = self.means[:, 1, :, :] + \
                self.coeffs[:, 0, :, :] * samples[:, 0, :, :]                     # B, M, W
        
        mean3 = self.means[:, 2, :, :]            
                                           # B, M, W
        mean4 = self.means[:, 3, :, :] + \
                self.coeffs[:, 1, :, :] * samples[:, 2, :, :]                     # B, M, W
        mean5 = self.means[:, 4, :, :] + \
                self.coeffs[:, 2, :, :] * samples[:, 2, :, :] + \
                self.coeffs[:, 3, :, :] * samples[:, 3, :, :]                     # B, M, W
        
        mean6 = self.means[:, 5, :, :] + \
                self.coeffs[:, 4, :, :] * samples[:, 2, :, :] + \
                self.coeffs[:, 5, :, :] * samples[:, 3, :, :] + \
                self.coeffs[:, 6, :, :] * samples[:, 4, :, :]                             # B, M, W
        mean7 = self.means[:, 6, :, :] + \
                self.coeffs[:, 7, :, :] * samples[:, 2, :, :] + \
                self.coeffs[:, 8, :, :] * samples[:, 3, :, :] + \
                self.coeffs[:, 9, :, :] * samples[:, 4, :, :] + \
                self.coeffs[:, 10, :, :] * samples[:, 5, :, :]                   
        mean8 = self.means[:, 7, :, :] + \
                self.coeffs[:, 11, :, :] * samples[:, 2, :, :] + \
                self.coeffs[:, 12, :, :] * samples[:, 3, :, :] + \
                self.coeffs[:, 13, :, :] * samples[:, 4, :, :] + \
                self.coeffs[:, 14, :, :] * samples[:, 5, :, :] + \
                self.coeffs[:, 15, :, :] * samples[:, 6, :, :]     


        mean1 = mean1.unsqueeze(1)                          # B, 1, M, W
        mean2 = mean2.unsqueeze(1)                          # B, 1, M, W
        mean3 = mean3.unsqueeze(1)                          # B, 1, M, W
        mean4 = mean4.unsqueeze(1)                          # B, 1, M, W
        mean5 = mean5.unsqueeze(1)                          # B, 1, M, W
        mean6 = mean6.unsqueeze(1)                          # B, 1, M, W
        mean7 = mean7.unsqueeze(1)                          # B, 1, M, W
        mean8 = mean8.unsqueeze(1)                          # B, 1, M, W
        means = torch.cat([mean1, mean2, mean3,mean4, mean5, mean6,mean7, mean8], dim=1)     # B, 8, M, W
        centered = samples - means                          # B, 8, M, W

        inv_stdv = torch.exp(- self.log_scales)
        plus_in = inv_stdv * (centered + 1. / self.max_val)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_stdv * (centered - 1. / self.max_val)
        cdf_min = torch.sigmoid(min_in)
        log_cdf_plus = plus_in - F.softplus(plus_in)
        log_one_minus_cdf_min = - F.softplus(min_in)
        cdf_delta = cdf_plus - cdf_min
        mid_in = inv_stdv * centered
        log_pdf_mid = mid_in - self.log_scales - 2. * F.softplus(mid_in)

        log_prob_mid_safe = torch.where(cdf_delta > 1e-5,
                                        torch.log(torch.clamp(cdf_delta, min=1e-10)),
                                        log_pdf_mid - np.log(self.max_val / 2))

        # the original implementation uses samples > 0.999, this ignores the largest possible pixel value (255)
        # which is mapped to 0.9922
        log_probs = torch.where(samples < -0.999, log_cdf_plus, torch.where(samples > 0.99, log_one_minus_cdf_min,
                                                                            log_prob_mid_safe))   # B, 8, M, W

        log_probs = torch.sum(log_probs, 1) + F.log_softmax(self.logit_probs, dim=1)  # B, M, W
        if self.focal:
            probs = torch.exp(log_probs)
            loss = (1-probs)*torch.log(probs)
            return torch.sum(loss, dim=1)
        else:
            return torch.logsumexp(log_probs, dim=1)                                      # B, W

    def sample(self, t=1.):
        # gumbel from uniform
        gumbel = -torch.log(- torch.log(torch.Tensor(self.logit_probs.size()).uniform_(1e-5, 1. - 1e-5).cuda()))  # B, M, W
        
        # select by using gumbel-argmax best value for pixel
        sel = one_hot(torch.argmax(self.logit_probs / t + gumbel, 1), self.num_mix, dim=1)          # B, M, W
        sel = sel.unsqueeze(1)                                                                 # B, 1, M, W  
        
        # select logistic parameters
        means = torch.sum(self.means * sel, dim=2)                                             # B, 8, W
        log_scales = torch.sum(self.log_scales * sel, dim=2)                                   # B, 8, W
        coeffs = torch.sum(self.coeffs * sel, dim=2)                                           # B, 8, W

        # cells from logistic & clip to interval
        # we don't actually round to the nearest 8bit value when sampling
        u = torch.Tensor(means.size()).uniform_(1e-5, 1. - 1e-5).cuda()                        # B, 8, W
        x = means + torch.exp(log_scales) / t * (torch.log(u) - torch.log(1. - u))             # B, 8, W
        
        x0 = torch.clamp(x[:, 0, :], -1, 1.)                                                # B, W
        x1 = torch.clamp(x[:, 1, :] + coeffs[:, 0, :] * x0, -1, 1)                       # B, W

        x2 = torch.clamp(x[:, 2, :], -1, 1)  # B, W
        x3 = torch.clamp(x[:, 3, :] + coeffs[:,1,:]*x2, -1, 1.)                                                # B, W
        x4 = torch.clamp(x[:, 4, :] + coeffs[:, 2, :] * x2 + coeffs[:,3,:]*x3, -1, 1)                       # B, W
        x5 = torch.clamp(x[:, 5, :] + coeffs[:, 4, :] * x2 + coeffs[:,5,:]*x3 + coeffs[:,6,:]*x4, -1, 1)  # B, W
        x6 = torch.clamp(x[:, 6, :] + coeffs[:, 7, :] * x2 + coeffs[:,8,:]*x3 + coeffs[:,9,:]*x4 + coeffs[:,10,:]*x5, -1, 1.)                                                # B, W
        x7 = torch.clamp(x[:, 7, :] + coeffs[:, 11, :] * x2 + coeffs[:,12,:]*x3 + coeffs[:,13,:]*x4 + coeffs[:,14,:]*x5 + coeffs[:,15,:]*x6, -1, 1)                       # B, W

        x0 = x0.unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        x3 = x3.unsqueeze(1)
        x4 = x4.unsqueeze(1)
        x5 = x5.unsqueeze(1)
        x6 = x6.unsqueeze(1)
        x7 = x7.unsqueeze(1)

        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], 1)
        x = x / 2. + 0.5
        return x
