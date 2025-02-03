import torch
import hamiltorch

def set_seeds(chainId):
    if chainId == 0:
        torch.random.manual_seed(3523421)
        hamiltorch.set_random_seed(3523421)
    elif chainId == 1:
        torch.random.manual_seed(989)
        hamiltorch.set_random_seed(989)
    else:
        assert(False)
        
def get_samples_as_dict_and_transform(model, all_samples):
    name_to_params = {}

    total_nr_mcmc_samples = len(all_samples)

    for sample_id, params in enumerate(all_samples):
        i = 0
        for name, model_param in model.named_parameters():
            if name not in name_to_params:
                name_to_params[name] = torch.zeros((total_nr_mcmc_samples, model_param.numel()))
            
            name_to_params[name][sample_id] = params[i: (i + model_param.numel())]

            if name in model.all_transforms:
                transformation = model.all_transforms[name]
                name_to_params[name][sample_id] = transformation(name_to_params[name][sample_id])

            i += model_param.numel()

    return name_to_params

def merge_chains_as_tensor(all_samples):

    NR_CHAINS = len(all_samples)
    assert(NR_CHAINS >= 2 and NR_CHAINS <= 4)

    all_samples_as_tensor = {}

    for name in all_samples[0]:
        for chain_id, samples_one_chain in enumerate(all_samples):
            if name not in all_samples_as_tensor:
                nr_samples = samples_one_chain[name].shape[0]
                nr_dims = samples_one_chain[name].shape[1]
                all_samples_as_tensor[name] = torch.zeros((NR_CHAINS, nr_samples, nr_dims))
            all_samples_as_tensor[name][chain_id] = samples_one_chain[name]

    return all_samples_as_tensor


def get_log_prob_func(model, X, y):
    fmodel = hamiltorch.util.make_functional(model)

    def log_prob_func(params):
        params_unflattened = hamiltorch.util.unflatten(model, params)

        name_to_params = {}
        i = 0
        for name, model_param in model.named_parameters():
            name_to_params[name] = params[i: (i + model_param.numel())]
            i += model_param.numel()
        
        logPrior = model.getLogPrior(name_to_params)

        mean_predictions = fmodel(X, params=params_unflattened)
        logLikelihood = model.getLogLikelihood(name_to_params, mean_predictions, y)
        
        return logPrior + logLikelihood

    return log_prob_func