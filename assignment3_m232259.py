import generate_data
import numpy as np
import torch

import matplotlib.pyplot as plt

from torch.distributions.normal import Normal
import hamiltorch
import hamiltorch.util
import pyro

import mcmc_utils


# "nu" (roughly) corresponds to number of prior observations
# "sSquare" (roughly) corresponds to prior variance
# log prob up to constant
def log_prob_scaledInvGamma(nu, sSquare, theta):
    log_prob = - ((nu / 2.0) + 1.0) * torch.log(theta)
    log_prob -= nu * sSquare / (2 * theta)
    return log_prob


class LinearRegressionModel(torch.nn.Module):

    def __init__(self, p):
        m = p // 2
        super().__init__()
        self.linear = torch.nn.Linear(p, m, bias = True)
        self.beta = torch.nn.Parameter(torch.randn(m))
        self.tau = torch.nn.Parameter(torch.randn(1))
        self.sigmaSquared_noise_unconstrained = torch.nn.parameter.Parameter(torch.rand(1)) # sigmasquare
        
        self.all_transforms = {}
        self.all_transforms["sigmaSquared_noise_unconstrained"] = torch.distributions.transforms.ExpTransform()
        

    def forward(self, x):
        linear_output = self.linear(x) 
        activated_output = torch.relu(linear_output) 
        beta_output = torch.matmul(activated_output, self.beta)
        final_output = beta_output + self.tau
        return final_output


    def getLogLikelihood(self, name_to_params, mean_predictions, y):
        sigmaSquared_noise = self.getTransformedParameter(name_to_params, "sigmaSquared_noise_unconstrained") 
        return torch.sum(Normal(mean_predictions, torch.ones_like(mean_predictions) * torch.sqrt(sigmaSquared_noise)).log_prob(y)) # N() getLikelihood


    def getLogPrior(self, name_to_params):

        # print("name_to_params = ", name_to_params)
        # assert(False)

        # ******** prior p(beta) ************
        logPrior = torch.zeros((1))
        sigmaSquared_coeff = torch.ones((1))
        for name, data in name_to_params.items():
            if not name.startswith("sigmaSquared_noise_unconstrained"):   
                # for all parameters other than "sigmaSquared_noise_unconstrained" we assume a Normal Distribution N(0, 1) as prior
                logPrior += torch.sum(Normal(torch.zeros_like(data),  torch.ones_like(data) * torch.sqrt(sigmaSquared_coeff)).log_prob(data))
        
        
        # ******** prior p(sigma^2) ************ / inverse-chi-squared
        sigmaSquared_noise = self.getTransformedParameter(name_to_params, "sigmaSquared_noise_unconstrained") 
        logPrior += log_prob_scaledInvGamma(nu = 1.0, sSquare = 1.0, theta = sigmaSquared_noise) 
        
        # ******** add all log det jacobians ************
        # To understand why we need the log_abs_det_jacobian, see e.g. the following:
        # https://en.wikibooks.org/wiki/Probability/Transformation_of_Probability_Densities
        # https://pytorch.org/docs/stable/_modules/torch/distributions/transforms.html#ExpTransform
        for name, data in name_to_params.items():
            if name in model.all_transforms:
                transformation = model.all_transforms[name]
                logPrior += transformation.log_abs_det_jacobian(data, _)

        return logPrior


    def getTransformedParameter(self, name_to_params, name):
        transformation = self.all_transforms[name]
        return transformation(name_to_params[name])

# ensure that we always get the same data D
np.random.seed(3523421) 


n = 20
p = 10

true_beta = generate_data.get_true_beta(p)

print("true_beta = ", true_beta)

X, y, _ = generate_data.linear_regression_data(n, true_beta)

X = torch.from_numpy(X).float()
y = torch.from_numpy(y).float()


NUM_SAMPLES_TOTAL = 1000  # number of MCMC samples
NUM_SAMPLES_2 = 2000

# Parameters for Hamiltorch:
STEP_SIZE = 0.01
NUM_STEPS_PER_SAMPLE = 5
SAMPLER=hamiltorch.Sampler.HMC_NUTS
INTEGRATOR=hamiltorch.Integrator.IMPLICIT

all_samples = []
all_samples_2 = []

for chainId in range(2):
    mcmc_utils.set_seeds(chainId) # use different seeds for each chain in order to create different initial values of the parameters

    model = LinearRegressionModel(p)
    initial_parameters = hamiltorch.util.flatten(model).clone()

    print("initial_parameters = ", initial_parameters)
    log_prob_func = mcmc_utils.get_log_prob_func(model, X, y) # this creates the log-joint probability as a function of the parameters theta
    all_samples_one_chain = hamiltorch.sample(log_prob_func=log_prob_func,burn = int(0.5 * NUM_SAMPLES_TOTAL),params_init=initial_parameters,  
                                                num_samples=NUM_SAMPLES_TOTAL, step_size=STEP_SIZE, num_steps_per_sample=NUM_STEPS_PER_SAMPLE, 
                                                sampler=SAMPLER, integrator=INTEGRATOR)
    
    all_samples_one_chain_dict = mcmc_utils.get_samples_as_dict_and_transform(model, all_samples_one_chain)

    all_samples.append(all_samples_one_chain_dict)

    samples_one_chain = hamiltorch.sample(log_prob_func=log_prob_func,burn = int(0.5 * NUM_SAMPLES_2),params_init=initial_parameters,  
                                                num_samples=NUM_SAMPLES_2, step_size=STEP_SIZE, num_steps_per_sample=NUM_STEPS_PER_SAMPLE, 
                                                sampler=SAMPLER, integrator=INTEGRATOR)
    
    samples_one_chain_dict = mcmc_utils.get_samples_as_dict_and_transform(model, samples_one_chain)

    all_samples_2.append(samples_one_chain_dict)


all_samples_as_tensor = mcmc_utils.merge_chains_as_tensor(all_samples)
samples_as_tensor = mcmc_utils.merge_chains_as_tensor(all_samples_2)

# print out statistics of the posterior for each parameters and also the MCMC diagnostics n-eff and R-hat
pyro.infer.mcmc.util.print_summary(all_samples_as_tensor, prob=0.9, group_by_chain=True)
pyro.infer.mcmc.util.print_summary(samples_as_tensor, prob=0.9, group_by_chain=True)


# ******** plot of samples from joint posterior (beta[0], beta[1]) ***********
color_list = ["b", "g"]

for chain_id in range(len(color_list)):
    beta_samples = all_samples[chain_id]["linear.weight"]
    plt.scatter(beta_samples[:,0], beta_samples[:,2], color = color_list[chain_id], alpha=0.5)

plt.xlabel("beta[0]")
plt.ylabel("beta[2]")
plt.show()

# ******** plot of samples from marginal posteriors of beta[j], for j = 1..p  ***********
beta_samples = all_samples[0]["linear.weight"]

print("beta_samples = ", beta_samples)
print(beta_samples.shape)

for chain_id in range(len(color_list)):
    beta_samples_2 = all_samples_2[chain_id]["linear.weight"]
    plt.scatter(beta_samples_2[:,0], beta_samples_2[:,2], color = color_list[chain_id], alpha=0.5)

beta_samples_2 = all_samples_2[0]["linear.weight"]

print("beta_samples = ", beta_samples_2)
print(beta_samples_2.shape)

n_bins = 30
fig, axs = plt.subplots(1, p)
for j in range(p):
    axs[j].hist(beta_samples[:,j], bins=n_bins)
    axs[j].set_title(f"beta[{j}]")
plt.show()

# Increasing the number of samples from 1000 to 2000 per chain improves the quality of the posterior samples 
# by reducing Monte Carlo error.
# With more samples, the effect of autocorrelation is mitigated, and the effective sample size (ESS) increases, 
# leading to more independent samples from the posterior.
# The r_hat for both chain is more than 1.1, which indicates that the chains have not converged yet.


