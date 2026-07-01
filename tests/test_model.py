import torch
from aurum_genre.model import ShortChunkCNN


def test_forward_output_shape():
    m = ShortChunkCNN(num_classes=11)
    mel = torch.randn(4, 1, 128, 188)   # B=4, 128 mel, ~3.7s frames
    out = m(mel)
    assert out.shape == (4, 11)


def test_output_are_logits_not_probabilities():
    m = ShortChunkCNN(num_classes=11)
    out = m(torch.randn(2, 1, 128, 188))
    # logits: not constrained to [0,1]
    assert (out < 0).any() or (out > 1).any()
