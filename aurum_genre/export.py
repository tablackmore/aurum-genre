"""Export the trained CNN to ONNX (opset 17, mel tensor input) + golden vectors.
The ONNX graph contains NO mel/STFT — feature extraction stays host-side."""
from __future__ import annotations
import numpy as np
import torch
from .model import ShortChunkCNN
from .mel import log_mel, mel_recipe, CHUNK_SAMPLES


def export_onnx(ckpt: str, out_onnx: str, golden_out: str, recipe_out: str) -> None:
    # weights_only=True: refuse to unpickle arbitrary objects (code-exec safety);
    # our checkpoint is only a state_dict + a list of root-label strings.
    blob = torch.load(ckpt, map_location="cpu", weights_only=True)
    roots = blob["roots"]
    model = ShortChunkCNN(num_classes=len(roots))
    model.load_state_dict(blob["state_dict"])
    model.eval()

    dummy = torch.randn(1, 1, 128, 188)
    torch.onnx.export(
        model, dummy, out_onnx, opset_version=17,
        input_names=["mel"], output_names=["logits"],
        dynamic_axes={"mel": {0: "batch", 3: "frames"}, "logits": {0: "batch"}},
    )

    # golden: a fixed waveform → its mel → onnx logits, for the Rust parity test
    torch.manual_seed(1234)
    wav = (torch.rand(1, CHUNK_SAMPLES) * 2 - 1)
    mel = log_mel(wav).unsqueeze(0)            # [1,1,128,T]
    import onnxruntime as ort
    sess = ort.InferenceSession(out_onnx, providers=["CPUExecutionProvider"])
    logits = sess.run(None, {"mel": mel.numpy()})[0]
    np.savez(golden_out, waveform=wav.numpy(), mel=mel.numpy(),
             logits=logits, roots=np.array(roots))
    with open(recipe_out, "w") as f:
        f.write(mel_recipe())
    print(f"wrote {out_onnx}, {golden_out}, {recipe_out}")
