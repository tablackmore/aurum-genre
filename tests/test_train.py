import torch
from aurum_genre.model import ShortChunkCNN
from aurum_genre.train import train_one_epoch

def test_loss_decreases_on_repeated_tiny_batch():
    torch.manual_seed(0)
    mel = torch.randn(6, 1, 128, 188)
    target = torch.zeros(6, 3); target[:, 0] = 1.0
    loader = [(mel, target)] * 8
    model = ShortChunkCNN(num_classes=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    first = train_one_epoch(model, loader, optim, "cpu")
    last = None
    for _ in range(5):
        last = train_one_epoch(model, loader, optim, "cpu")
    assert last < first

def test_train_one_epoch_with_mixup_runs_and_is_finite():
    torch.manual_seed(0)
    mel = torch.randn(6, 1, 128, 188)
    target = torch.zeros(6, 3); target[:, 0] = 1.0
    loader = [(mel, target)] * 4
    model = ShortChunkCNN(num_classes=3)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, optim, "cpu", mixup_alpha=0.2)
    assert loss == loss and loss >= 0.0        # not NaN, valid loss
