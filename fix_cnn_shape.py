import pathlib
p = pathlib.Path("experiments/run_training.py")
t = p.read_text(encoding="utf-8")

# 1. make_model must forward image_shape to the torch CNN
t = t.replace(
    "def make_model(arch, input_dim, num_classes, hidden, use_torch, image_shape=None):",
    "def make_model(arch, input_dim, num_classes, hidden, use_torch, image_shape=None):"
)  # signature already has image_shape; ensure run_once passes it

# 2. run_once: derive image_shape for cnn and pass it in
t = t.replace(
    "    model = make_model(arch, input_dim, num_classes, hidden, use_torch)",
    "    img_shape = (3, 32, 32) if arch == \"cnn\" else None\n"
    "    model = make_model(arch, input_dim, num_classes, hidden, use_torch, image_shape=img_shape)"
)

p.write_text(t, encoding="utf-8")
print("patched run_training.py")
print("make_model call now passes image_shape:", "img_shape" in t)
