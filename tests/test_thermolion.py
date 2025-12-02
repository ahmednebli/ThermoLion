import torch

from thermolion import ThermoLion


def test_thermolion_runs_and_decreases_loss():
    torch.manual_seed(0)

    model = torch.nn.Linear(10, 1)
    x = torch.randn(128, 10)
    y = torch.randn(128, 1)

    criterion = torch.nn.MSELoss()
    optimizer = ThermoLion(model.parameters(), lr=1e-2)

    with torch.no_grad():
        initial_loss = criterion(model(x), y).item()

    for _ in range(25):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        final_loss = criterion(model(x), y).item()

    assert final_loss < initial_loss
