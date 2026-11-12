`model.py` and `engram.py` are copied unchanged from `deepseek-ai/DeepSeek-V4.1-Flash-1017`, in the `inference/` directory at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`. They are MIT-licensed; the license is in `LICENSE`. `model.py` has sha256 `e5decd76fe70fd56ef2f10124d245c6f64fb6a0003cc9068a1d3182a168b8d2d`.

The CPU tests in `tests/test_runpod_v41_stream.py` run them at toy size. They swap in `scripts/runpod/v41_reap/torch_kernel.py` for the reference's tilelang `kernel.py`, and stub out the modules used only for vision.
