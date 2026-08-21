"""Execute the README's code block, so that API drift turns CI red.

The block is executable as written except for the region between the BEGIN USER
and END USER markers, which is replaced here by a small concrete model and two
loaders. Everything else -- the imports, the config construction, the optimizer
signature and the training loop -- is run verbatim.
"""

from __future__ import annotations

import pathlib
import re
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]

PRELUDE = """
model = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.Tanh(),
                            torch.nn.Linear(12, 3))
loss_fn = torch.nn.CrossEntropyLoss()
_X = torch.randn(64, 8)
_Y = torch.randint(0, 3, (64,))
_ds = torch.utils.data.TensorDataset(_X, _Y)
data_loader = torch.utils.data.DataLoader(_ds, batch_size=16)
hg_loader = torch.utils.data.DataLoader(_ds, batch_size=16, drop_last=True)
"""


def extract() -> str:
    text = (ROOT / "README.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    if not blocks:
        sys.exit("README.md: no ```python block found")
    code = blocks[0]
    if "BEGIN USER" not in code or "END USER" not in code:
        sys.exit("README.md: the python block lost its BEGIN USER / END USER markers")
    head, rest = code.split("BEGIN USER", 1)
    _, tail = rest.split("END USER", 1)
    head = head.rsplit("\n", 1)[0]  # drop the marker's own comment line
    tail = tail.split("\n", 1)[1]
    return head + "\n" + PRELUDE + "\n" + tail


def main() -> None:
    code = extract()
    code = code.replace("for epoch in range(10):", "for epoch in range(1):")
    ns = {"__name__": "__readme__", "torch": torch}
    exec(compile(code, "README.md", "exec"), ns)  # noqa: S102
    model = ns["model"]
    assert all(torch.isfinite(p).all() for p in model.parameters()), (
        "README example produced non-finite parameters"
    )
    print("README example: OK")


if __name__ == "__main__":
    main()
