# ThermoLion

ThermoLion is a research optimizer for PyTorch that combines Lion-style
sign updates with variance-normalised (Adam-like) updates and an annealed
thermodynamic exploration term.

It is extracted from an experimental benchmark script and packaged as a
small, installable library so that you can `pip install thermolion` and
drop it into your own training code.

> **Status:** research / experimental. Expect to tune hyperparameters for
> your own workloads.

---

## Installation

### 1. From PyPI (recommended once released)

```bash
pip install thermolion
```

### 2. From source (this repository)

Clone the repo and install in editable mode:

```bash
git clone https://github.com/ahmednebli/ThermoLion.git
cd ThermoLion

# (optional) create and activate a virtualenv here

pip install -e ".[dev]"
```

This will install the `thermolion` package and its dependencies into your
current environment, including development tools like `pytest` if you use
the `.[dev]` extra.

---

## Usage

The optimizer follows the standard PyTorch `Optimizer` API, so you can
drop it into any existing training loop by swapping out your current
optimizer.

### Basic example

```python
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from thermolion import ThermoLion

# Dummy dataset
x = torch.randn(1024, 32)
y = torch.randint(0, 10, (1024,))
loader = DataLoader(TensorDataset(x, y), batch_size=128, shuffle=True)

model = nn.Sequential(
    nn.Linear(32, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

criterion = nn.CrossEntropyLoss()
optimizer = ThermoLion(model.parameters(), lr=1e-3, betas=(0.9, 0.99), temp_decay=0.99, weight_decay=0.01)

model.train()
for epoch in range(10):
    total_loss = 0.0
    for inputs, targets in loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch:02d} | Loss: {total_loss / len(loader):.4f}")
```

### Plugging into your existing benchmark script

If you already have a script that benchmarks multiple optimizers (Adam,
AdamW, RMSProp, Lion, MuAdam, Lookahead, SWATS, and SWA), you can usually
integrate ThermoLion by:

1. Importing the optimizer:

   ```python
   from thermolion import ThermoLion
   ```

2. Swapping the construction where you choose the optimizer, e.g.:

   ```python
   if opt_name == "ThermoLion":
       optimizer = ThermoLion(model.parameters(), lr=1e-3)
   ```

The rest of your training loop (forward, loss, backward, step) remains the
same.

---

## Hyperparameters

```python
ThermoLion(
    params,
    lr=1e-3,
    betas=(0.9, 0.99),
    temp_decay=0.99,
    weight_decay=0.01,
    eps=1e-8,
)
```

- **lr**: base learning rate. Start with the same order of magnitude that
  you would use for Adam on the same model (e.g. `1e-3` on vision models).

- **betas**: `(beta1, beta2)` for the first and second moment estimates.
  Defaults are tuned to behave similarly to Lion-style momentum and
  Adam-like variance tracking.

- **temp_decay**: multiplicative decay factor for the "temperature"
  that scales the stochastic exploration term. Values closer to 1.0 keep
  exploration active for longer; values like `0.95`–`0.99` are a reasonable
  range to experiment with.

- **weight_decay**: decoupled weight decay (AdamW-style). Set to `0.0`
  if you prefer to manage regularisation separately or use explicit L2
  terms in your loss.

- **eps**: small constant added to the denominator for numerical stability.

---

## Reproducing the original image benchmark

In the original script, ThermoLion was benchmarked on a collection of
image datasets (MNIST, CIFAR-10/100, SVHN, STL-10, and others) alongside
Adam, AdamW, RMSprop, Lion, MuAdam, Lookahead, SWATS, and SWA.

To reproduce a similar setup:

1. Factor your dataset/model code into a function that returns a
   `(dataloader, model)` pair (as you already did).
2. Add ThermoLion to the list of optimizers.
3. Instantiate it with the same API as in the basic usage example above.

You don't need any special hooks beyond the standard `optimizer.zero_grad()`
/ `loss.backward()` / `optimizer.step()` trio.

---

## Development

### Running tests

Install the dev dependencies and run `pytest`:

```bash
pip install -e ".[dev]"
pytest
```

---

## Publishing to PyPI (for maintainers)

1. Build the distribution:

   ```bash
   pip install build twine
   python -m build
   ```

   This creates `dist/thermolion-<version>.tar.gz` and a wheel file.

2. Upload to PyPI:

   ```bash
   twine upload dist/*
   ```

   For testing first, use `--repository testpypi` and install with
   `pip install -i https://test.pypi.org/simple thermolion`.

3. Tag the release in git and push:

   ```bash
   git tag v0.1.0
   git push --tags
   ```

---

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE)
for details.
