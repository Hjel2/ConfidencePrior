from src.experiments import ewc

epochs = 1
device = 'cpu'
n_tasks = 10
sample_size = 1000
lambd = 120.

print(f'{epochs=} {sample_size=} {lambd=}')
ewc(
    epochs=epochs,
    sample_size=sample_size,
    device=device,
    n_tasks=n_tasks,
    lambd = lambd,
)
