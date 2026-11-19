import subprocess
import sys

TRAIN_ONLY = ["albumentations", "torch", "timm", "segmentation_models_pytorch", "sklearn"]


def test_api_runtime_does_not_import_training_libraries():
    code = ("import sys; import backend.app, choleguard.inference; "
            f"print(','.join(m for m in {TRAIN_ONLY!r} if m in sys.modules))")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=__import__("pathlib").Path(__file__).resolve().parents[1],
                       env={**__import__("os").environ, "PYTHONPATH": "src:."})
    assert r.returncode == 0, r.stderr[-1500:]
    assert r.stdout.strip() == "", f"API pulls training-only libraries: {r.stdout.strip()}"
