from src.experiments import coreset_only

epochs = 5
coreset_method='random'
device='cpu'
n_tasks=10
c=1/2
logit_std=11

coreset_only(
    epochs=epochs,
    coreset_size=0,
    coreset_method=None,
    device=device,
    n_tasks=n_tasks,
    c=c,
    logit_std=logit_std,
)

for coreset_size in (200, 400, 1000, 2500, 5000):
    coreset_only(
        epochs=5,
        coreset_size=coreset_size,
        coreset_method=coreset_method,
        device=device,
        n_tasks=n_tasks,
        c=c,
        logit_std=logit_std,
    )

