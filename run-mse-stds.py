from src.experiments import coreset_vcl

epochs = 10
coreset_method='random'
device='cpu'
n_tasks=10
loss_fn='mse'
std=0.1
logit_std=0.3162


for std in (0.01, 0.03, 0.1, 0.3, 1):
    coreset_vcl(
        epochs=epochs,
        coreset_size=0,
        coreset_method=None,
        device=device,
        n_tasks=n_tasks,
        loss_fn=loss_fn,
        std=std,
        logit_std=logit_std,
    )
