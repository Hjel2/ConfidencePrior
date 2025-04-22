from src.experiments import coreset_vcl

epochs = 5
coreset_method='random'
device='cpu'
n_tasks=10
c=2/3


# 75, 80, 85, 90, 95, 99
for logit_std in (107,):# (4.20, 5.31, 7.13, 10.72, 21.45, 107):
    print(f'{logit_std=}')
    coreset_vcl(
        epochs=epochs,
        coreset_size=0,
        coreset_method=None,
        device=device,
        n_tasks=n_tasks,
        c=c,
        logit_std=logit_std,
    )
