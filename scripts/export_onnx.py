"""Export a CVS checkpoint to ONNX: image -> (seg_probs, calibrated cvs_probs)."""
import argparse

import numpy as np
import onnxruntime as ort
import torch

from choleguard.training import load_ckpt


class Exported(torch.nn.Module):
    def __init__(self, model, temperature):
        super().__init__()
        self.model = model
        self.register_buffer("t", torch.tensor(temperature, dtype=torch.float32))

    def forward(self, image):
        out = self.model(image)
        return out["seg_logits"].softmax(1), torch.sigmoid(out["cvs_logits"] / self.t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", nargs=2, type=int, default=[256, 448])
    a = ap.parse_args()
    model, ck = load_ckpt(a.ckpt, "cpu")
    net = Exported(model.eval(), ck.get("temperature", [1.0, 1.0, 1.0])).eval()
    x = torch.randn(1, 3, *a.size)
    torch.onnx.export(net, x, a.out, input_names=["image"], output_names=["seg_probs", "cvs_probs"],
                      opset_version=17, dynamo=False)
    with torch.no_grad():
        ref = [t.numpy() for t in net(x)]
    got = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})
    err = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    print(f"max |torch - onnx| = {err:.2e}")
    if err > 1e-3:
        raise SystemExit("ONNX export mismatch")


if __name__ == "__main__":
    main()
