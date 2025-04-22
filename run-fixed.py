from src.experiments import fixed_size_coreset

epochs = 10
coreset_method = 'random'
device = 'cpu'
n_tasks = 10
c=2/3
logit_std=10.7 # 90% accuracy prior


for coreset_size in (
    500,
    1000,
    2000,
    5000,
    10000,
    25000,
    50000,
):
    print(f'{coreset_size=}')
    fixed_size_coreset(
        epochs=epochs,
        coreset_size=coreset_size,
        coreset_method=coreset_method,
        device=device,
        n_tasks=n_tasks,
        c=c,
        logit_std=logit_std,
    )
