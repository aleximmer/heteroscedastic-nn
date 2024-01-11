from itertools import product
CRISPR_DATASETS = [
    'flow-cytometry-HEK293', 'survival-screen-A375', 'survival-screen-HEK293'
]
seeds = [seed for seed in range(1, 11)]
heads = ['natural', 'meanvar']
for seed, dataset in product(seeds, CRISPR_DATASETS):
    base_cmd = f'python run_uci_crispr_regression.py --seed {seed} --dataset {dataset} --config configs/crispr.yaml'
    # homoscedastic
    print(base_cmd, '--likelihood homoscedastic --method map')
    print(base_cmd, '--likelihood homoscedastic --method marglik')

    # heteroscedastic
    print(base_cmd, f'--likelihood heteroscedastic --method faithful --head gaussian')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.0')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.5')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 1.0')
    # Bayes (VI)
    vi_params = '--n_epochs 500 --lr 0.001 --lr_min 0.001 --optimizer Adam'
    print(base_cmd, f'--likelihood heteroscedastic --method vi {vi_params}')
    # Proposed Laplace approximation
    for head in heads:
        print(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')


