from itertools import product
UCI_DATASETS = [
    'boston-housing', 'concrete', 'energy', 'kin8nm','naval-propulsion-plant',
    'power-plant', 'wine-quality-red', 'yacht'
]

seeds = [seed for seed in range(1, 21)]
heads = ['natural', 'meanvar']
for seed, dataset in product(seeds, UCI_DATASETS):
    base_cmd = f'python run_uci_crispr_regression.py --seed {seed} --dataset {dataset} --config configs/uci.yaml'
    # homoscedastic
    print(base_cmd, '--likelihood homoscedastic --method map')
    print(base_cmd, '--likelihood homoscedastic --method marglik')

    # heteroscedastic
    print(base_cmd, f'--likelihood heteroscedastic --method faithful --head gaussian')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.0')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 0.5')
    print(base_cmd, f'--likelihood heteroscedastic --method betanll --head gaussian --beta 1.0')
    # Bayes (MCDO, VI)
    mcdo_vi_params = '--n_epochs 1000 --lr 0.001 --lr_min 0.001 --optimizer Adam'
    print(base_cmd, f'--likelihood heteroscedastic --method mcdropout {mcdo_vi_params}')
    print(base_cmd, f'--likelihood heteroscedastic --method vi {mcdo_vi_params}')
    # Proposed Laplace approximation
    for head in heads:
        print(base_cmd, f'--likelihood heteroscedastic --method map --head {head}')
        print(base_cmd, f'--likelihood heteroscedastic --method marglik --head {head}')


