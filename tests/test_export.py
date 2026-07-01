import numpy as np, torch, onnxruntime as ort
from aurum_genre.model import ShortChunkCNN
from aurum_genre.export import export_onnx

def test_export_matches_torch(tmp_path):
    torch.manual_seed(0)
    model = ShortChunkCNN(num_classes=5).eval()
    ckpt = tmp_path / "m.pt"
    torch.save({"state_dict": model.state_dict(),
                "roots": ["a","b","c","d","e"]}, ckpt)
    onnx_p = tmp_path / "genre.onnx"
    golden = tmp_path / "mel_golden.npz"
    recipe = tmp_path / "mel_recipe.txt"
    export_onnx(str(ckpt), str(onnx_p), str(golden), str(recipe))

    mel = torch.randn(1, 1, 128, 188)
    with torch.no_grad():
        t_out = model(mel).numpy()
    sess = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
    o_out = sess.run(None, {"mel": mel.numpy()})[0]
    assert np.max(np.abs(t_out - o_out)) < 1e-4
    assert recipe.read_text().count("n_mels=128") == 1
