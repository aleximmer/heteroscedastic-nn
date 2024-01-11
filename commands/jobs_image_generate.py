from itertools import product
seeds = [117, 189, 509, 832, 711]
heads = ['natural', 'meanvar']

# MNIST and FMNIST
datasets = ['mnist', 'fmnist']
het_flags = ['--het_noise label', '--het_noise rotation', '--het_noise neither']
for seed, dataset, hf in product(seeds, datasets, het_flags):
    base_cmd = f'python run_image_regression.py --seed {seed} --config configs/{dataset}.yaml {hf}'
    # homoscedastic
    print(base_cmd, '--likelihood homoscedastic --method map')
    print(base_cmd, '--likelihood homoscedastic --method marglik')

    # heteroscedastic
    print(base_cmd, f'--likelihood heteroscedastic --method faithful --head gaussian')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.0')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.5')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 1.0')
    # Bayes (MCDO, VI)
    mcdo_params = '--n_epochs 50 --lr 0.01 --lr_min 0.01 --optimizer Adam'
    print(base_cmd, f'--likelihood heteroscedastic --method mcdropout {mcdo_params}')
    vi_lr = 1e-2 if 'fmnist' in dataset else 1e-3
    vi_lr_min = vi_lr / 100
    vi_params = f'--lr {vi_lr} --lr_min {vi_lr_min} --optimizer Adam --vi-posterior-rho-init -3.0'
    print(base_cmd, f'--likelihood heteroscedastic --method vi {vi_params}')
    # Proposed Laplace approximation
    for head in heads:
        print(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')

