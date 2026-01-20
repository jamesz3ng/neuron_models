from pathlib import Path

import numpy as np

_DEFAULT_BASIS_FILE = Path(__file__).parent.parent / "output" / "basis_data.npz"


def inspect_basis():
    try:
        data = np.load(_DEFAULT_BASIS_FILE)
        print("\n=== BASIS DATA CONTENT ===")
        print(f"{'Key':<20} | {'Shape':<15} | {'Value/Preview'}")
        print("-" * 60)

        for key in data.files:
            val = data[key]

            # Format value for display
            if val.ndim == 0:
                # Scalar (single number)
                preview = f"{val}"
            elif val.size < 6:
                # Small array
                preview = f"{val}"
            else:
                # Large array (show start/end)
                preview = f"[{val.flatten()[0]:.3f}, {val.flatten()[1]:.3f} ...]"

            print(f"{key:<20} | {str(val.shape):<15} | {preview}")

    except FileNotFoundError:
        print(f"Error: {_DEFAULT_BASIS_FILE} not found. Did you run ssds_model.py?")


if __name__ == "__main__":
    inspect_basis()
