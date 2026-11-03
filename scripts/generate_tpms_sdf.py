#!/usr/bin/env python3
"""Generate implicit TPMS fields in the input format expected by `shellhomo`.

Each field is sampled at the cell centres of a regular grid on the unit cube,
x_i = (i + 0.5) / res, with `ij` indexing (x is the slowest axis), and saved as a
flattened float64 `.npy` array of length res^3. The solver expects res = 257.

Example:
    python scripts/generate_tpms_sdf.py --out data/ --names Primitive Gyroid
"""
import argparse
import os

import numpy as np

c, s = np.cos, np.sin

# Each entry maps 2*pi-scaled coordinates (x, y, z) to the implicit function value.
TPMS = {
    "Primitive": lambda x, y, z: c(x) + c(y) + c(z),
    "DoublePrimitive": lambda x, y, z: 0.5 * (c(x) * c(y) + c(y) * c(z) + c(z) * c(x))
    + 0.2 * (c(2 * x) + c(2 * y) + c(2 * z)),
    "Diamond": lambda x, y, z: s(x) * s(y) * s(z) + s(x) * c(y) * c(z)
    + c(x) * s(y) * c(z) + c(x) * c(y) * s(z),
    "ComplementaryD": lambda x, y, z: c(3 * x + y) * c(z) - s(3 * x - y) * s(z)
    + c(x + 3 * y) * c(z) + s(x - 3 * y) * s(z) + c(x - y) * c(3 * z) - s(x + y) * s(3 * z),
    "DoubleDiamond": lambda x, y, z: 0.5 * (s(x) * s(y) + s(y) * s(z) + s(x) * s(z))
    + 0.5 * c(x) * c(y) * c(z),
    "DPrime": lambda x, y, z: 0.5 * (c(x) * c(y) * c(z) + c(x) * s(y) * s(z) + s(x) * c(y) * s(z)
                                     + s(x) * s(y) * c(z))
    - 0.5 * (s(2 * x) * s(2 * y) + s(2 * y) * s(2 * z) + s(2 * z) * s(2 * x)),
    "Gyroid": lambda x, y, z: c(x) * s(y) + c(y) * s(z) + c(z) * s(x),
    "GyroidPrime": lambda x, y, z: s(2 * x) * c(y) * s(z) + s(2 * y) * c(z) * s(x)
    + s(2 * z) * c(x) * s(y),
    "DoubleGyroid": lambda x, y, z: 2.75 * (s(2 * x) * s(z) * c(y) + s(2 * y) * s(x) * c(z)
                                            + s(2 * z) * s(y) * c(x))
    - (c(2 * x) * c(2 * y) + c(2 * y) * c(2 * z) + c(2 * z) * c(2 * x)),
    "KarcherK": lambda x, y, z: 0.3 * (c(x) + c(y) + c(z))
    + 0.3 * (c(x) * c(y) + c(y) * c(z) + c(z) * c(x)) - 0.4 * (c(2 * x) + c(2 * y) + c(2 * z)),
    "OctoCubic": lambda x, y, z: 0.6 * (c(x) * c(y) + c(y) * c(z) + c(z) * c(x))
    - 0.4 * (c(x) + c(y) + c(z)),
    "Lidinoid": lambda x, y, z: 0.5 * (s(2 * x) * c(y) * s(z) + s(2 * y) * c(z) * s(x)
                                       + s(2 * z) * c(x) * s(y))
    - 0.5 * (c(2 * x) * c(2 * y) + c(2 * y) * c(2 * z) + c(2 * z) * c(2 * x)),
    "Neovius": lambda x, y, z: 3.0 * (c(x) + c(y) + c(z)) + 4.0 * c(x) * c(y) * c(z),
    "IWP": lambda x, y, z: 2.0 * (c(x) * c(y) + c(y) * c(z) + c(z) * c(x))
    - (c(2 * x) + c(2 * y) + c(2 * z)),
    "FRD": lambda x, y, z: 4.0 * c(x) * c(y) * c(z)
    - (c(2 * x) * c(2 * y) + c(2 * y) * c(2 * z) + c(2 * z) * c(2 * x)),
    "FischerKochS": lambda x, y, z: c(2 * x) * s(y) * c(z) + c(x) * c(2 * y) * s(z)
    + s(x) * c(y) * c(2 * z),
    "FischerKochY": lambda x, y, z: s(x) * s(y) * s(z) + c(x) * c(y) * c(z) + s(2 * x) * s(y)
    + c(x) * s(2 * y) + s(x) * s(2 * z) + s(2 * x) * c(z) + s(2 * y) * s(z) + c(y) * s(2 * z),
}


def generate(name: str, res: int) -> np.ndarray:
    coords = (np.arange(res, dtype=np.float64) + 0.5) / res
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    field = TPMS[name](2 * np.pi * x, 2 * np.pi * y, 2 * np.pi * z)
    return field.astype(np.float64).reshape(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="data", help="output directory (default: data)")
    parser.add_argument("--res", type=int, default=257, help="samples per axis (the solver expects 257)")
    parser.add_argument("--names", nargs="+", default=list(TPMS), choices=list(TPMS), metavar="NAME",
                        help="surfaces to generate (default: all). Choices: " + ", ".join(TPMS))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for name in args.names:
        path = os.path.join(args.out, f"{name}_{args.res}.npy")
        np.save(path, generate(name, args.res))
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
