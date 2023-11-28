import torch

class Distribution:
    def __init__(self, name, coeffs, means, variances):
        if sum(coeffs) != 1:
            raise ValueError('Mixture coefficients do not sum to 1')
        if len(set([len(coeffs), len(means), len(variances)])) != 1:
            raise ValueError('Lengths of parameters must be the same')

        self.name = name
        self.num = len(coeffs)
        self.coeffs = coeffs
        self.means = means
        self.variances = variances

        self.gaussians = []
        for i in range(self.num):
            dist = torch.distributions.normal.Normal(self.means[i], self.variances[i])
            self.gaussians.append(dist)
    
    def prob(self, x):
        retval = 0

        for i in range(self.num):
            retval += self.coeffs[i] * torch.exp(self.gaussians[i].log_prob(x))

        return retval
    
    def log_prob(self, x):
        return torch.log(self.prob(x))
    
    def score(self, x):
        x_copy = x.detach()
        x_copy.requires_grad_()

        log_prob = self.log_prob(x_copy)
        log_prob.backward()

        return x_copy.grad
    
    def map(self, mode, inputs):
        if mode == 'prob':
            fn = self.prob
        elif mode == 'logprob':
            fn = self.log_prob
        elif mode == 'score':
            fn = self.score
        else:
            raise ValueError('Invalid mode')

        outputs = [fn(x) for x in inputs]

        return outputs
    
    def sample(self):
        cat_dist = torch.distributions.categorical.Categorical(probs=torch.tensor(self.coeffs))
        sampled_gaussian = self.gaussians[cat_dist.sample()]

        return sampled_gaussian.sample()
    
    def generate_samples(self, num_samples):
        l = []

        for i in range(num_samples):
            l.append(self.sample())

        return l

    def generate_training_data(self, mode, num_samples):
        samples = self.generate_samples(num_samples)
        outputs = self.map(mode, samples)

        sample_data = torch.tensor([[sample] for sample in samples])
        output_data = torch.tensor([[output] for output in outputs])

        return sample_data, output_data