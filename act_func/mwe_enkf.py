from torch.nn.functional import one_hot
import torch


def enkf_lstsq(ens, model_out, obs, gamma, batch_s, ensemble_size):
    for i in range(batch_s):
        g_tmp = model_out[:, :, i]
        Cpp = torch.tensordot(
            (g_tmp - g_tmp.mean(0)), (g_tmp - g_tmp.mean(0)), dims=([0], [0])) / ensemble_size
        Cup = torch.tensordot(
            (ens - ens.mean(0)), (g_tmp - g_tmp.mean(0)), dims=([0], [0])) / ensemble_size
        new_ens = torch.mm(Cup, torch.lstsq(
            (obs[i] - g_tmp).t(), Cpp+gamma)[0]).t() + ens
        return new_ens


def enkf_cholesky(ens, model_out, obs, gamma, batch_s, ensemble_size):
    mo_mean = model_out.mean(0)
    Cpp = torch.einsum("ijk, ilk -> kjl", model_out -
                       mo_mean, model_out - mo_mean) / ensemble_size
    Cup = torch.einsum("ij, ilk -> kjl", ens - ens.mean(0),
                       model_output - mo_mean) / ensemble_size
    tmp = torch.empty_like(Cpp)
    loss = torch.empty(batch_size, ensemble_size, gamma.shape[0])
    for i in range(batch_s):
        tmp[i] = torch.cholesky_inverse(Cpp[i] + gamma)
        loss[i] = obs[i] - model_out[:, :, i]  
    # loss = (-1 * model_out + obs.reshape(gamma.shape[0], -1)
    #         ).reshape(-1, ensemble_size, gamma.shape[0])
    mm = torch.matmul(loss, tmp)
    # new_ens = torch.matmul(Cup, mm) + ens
    new_ens = torch.einsum('ijk, ilk -> lj', Cup, mm) + ens
    return new_ens


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    batch_size = 64
    ensemble_size = 5000
    gamma = torch.eye(10, device=device) * 0.01
    ensembles = torch.randn(ensemble_size, 18050, device=device)
    model_output = torch.randn(ensemble_size, 10, batch_size, device=device)
    observations = torch.randint(
        low=0, high=10, size=(batch_size,), device=device)
    observations = one_hot(observations)
    ensembles_lst = enkf_lstsq(ensembles, model_output,
                               observations, gamma, batch_size, ensemble_size)
    print(ensembles_lst)
    ensembles_chol = enkf_cholesky(ensembles, model_output,
                                   observations, gamma, batch_size,
                                   ensemble_size)
    print(ensembles_chol)
    print(torch.allclose(ensembles_lst, ensembles_chol))
