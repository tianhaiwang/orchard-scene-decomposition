# From 3D Gaussian Splatting to Simulation-Ready Orchard Digital Twins: An Instance-Aware Scene Decomposition Pipeline

## Installation

Use Python 3.12 in an isolated environment. The code was tested with Python 3.12.11 and the following dependencies:

| Dependency | Version |
|---|---|
| NumPy | 2.3.3 |
| SciPy | 1.16.2 |
| scikit-learn | 1.7.2 |
| Open3D | 0.19.0 |
| plyfile | 1.1.2 |
| pwlf | 2.5.2 |

From the repository root, install the package and its dependencies:

```sh
python -m pip install .
orchard-scene-decomposition --version
```

Optional components:

| Component | Install command |
|---|---|
| Qt workbench (PyQt5 5.x) | `python -m pip install ".[gui]"` |
| Cloth-simulation ground filter (1.1.7) | `python -m pip install ".[csf]"` |

## Data

[Download the dataset](https://drive.google.com/drive/folders/1_UDSANFCKjMa2aYNF0IYrMDe5LpP8xnp?usp=sharing). Place the downloaded PLY files in `data/` at the repository root, for example `data/scene_01.ply`. Each PLY vertex represents one Gaussian.

| File | Orchard | Gaussians |
|---|---|---:|
| `scene_01.ply` | Cherry, Cilah | 1,795,570 |
| `scene_02.ply` | Cherry, Cilah | 1,264,596 |
| `scene_03.ply` | Cherry, Lapin | 1,257,256 |
| `scene_04.ply` | Cherry, Robin | 2,079,993 |
| `scene_05.ply` | Cherry, Robin | 1,645,382 |
| `scene_06.ply` | Cherry, Santina | 1,570,710 |
| `scene_07.ply` | Cherry, Santina | 1,283,302 |
| `scene_08.ply` | Cherry, Skeena | 1,036,077 |
| `scene_09.ply` | Cherry, Skeena | 643,023 |
| `scene_10.ply` | Peach | 2,397,148 |

The dataset README provides attribute descriptions and SHA-256 checksums.

## License

[MIT](LICENSE). Dependencies retain their own licenses, including GPLv3-or-later for plyfile and GPLv3/commercial for PyQt5.

Reference for point-color conversion: [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/utils/sh_utils.py).
