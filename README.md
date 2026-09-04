# NeuSOGA

![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Status: Research Prototype](https://img.shields.io/badge/status-research%20prototype-blue.svg)
![arXiv Submission](https://img.shields.io/badge/arXiv-submission%20context-b31b1b.svg)

**NeuSOGA** (**Neu**ro-**S**ymbolic Geometric **A**bstraction) is a research prototype for transforming raw observations into compact symbolic mathematical descriptions through **topology-guided geometric abstraction**.

**Keywords:** neuro-symbolic learning, geometric abstraction, topology-guided segmentation, symbolic representation learning, 3D perception, reproducible research, research prototype, arXiv-linked submission

> Core idea: **Observation → Topology → Geometry → Symbol**
<p align="center">
  <img src="assets/figures/Neuro-symbolic-pipeline.jpg" widthcenter">
  <em>Overview of the NeuSOGA framework.</em>
</p>


This repository is prepared as an **academic companion repository** for an arXiv-linked submission. The current public branch focuses on **clarity, reproducibility, and visibility**: it provides a lightweight runnable demo and supporting documentation while intentionally excluding heavyweight benchmark assets.


## Research motivation

Many modern perception systems stop at latent features or task-specific predictions. NeuSOGA instead targets **explicit symbolic structure**: observations are first organized topologically, then abstracted geometrically, and finally distilled into concise symbolic forms that can be interpreted and reused downstream.

The repository therefore emphasizes:

- interpretable structure discovery from observations,
- topology-aware segmentation before symbolic abstraction,
- concise symbolic descriptions of geometry,
- reproducible prototype behavior for research readers.

## What is included here

This branch currently contains:

- a research-facing `README.md`,
- a lightweight reproducibility demo at `demo/neusoga_demo.py`,
- a minimal `requirements-demo.txt` describing the demo dependency footprint.

This branch **does not** bundle large external assets such as:

- SAM checkpoints,
- ModelNet40 data archives,
- full training logs,
- paper figures,
- benchmark-ready experiment pipelines.

Those resources must be downloaded separately by the user.

## Pipeline overview

The runnable demo follows the same high-level prototype story:

1. **Observation (O)**  
Acquire geometric observations from point clouds, projected views,
or segmented optical inputs.

2. **Topology (T)**  
Extract topology-aware structural cores and internal void nodes
using Euclidean Distance Transforms.

3. **Geometry (G)**  
Perform topology-guided segmentation and adaptive multi-scale
abstraction to generate sparse geometric control polygons.

4. **Symbol (S)**  
Convert control polygons into Implicit Area Spline representations,
yielding explicit symbolic mathematical models.

The included demo is intentionally lightweight and CPU-friendly. It is meant to **illustrate the representation pipeline**, not to claim the full empirical scope of the accompanying research submission.


## Installation and setup

### 1. Clone the repository

```bash
git clone https://github.com/QL-UoHull/NeuSOGA.git
cd NeuSOGA
```

### 2. Create a Python environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-demo.txt
```

### 3. External resources you may need

- **ModelNet40**: not bundled in the repository. The demo can optionally download and extract the official archive when requested.
- **SAM checkpoints**: not bundled in the repository. If your extended workflow depends on Segment Anything, download the checkpoint separately and pass its path explicitly.
- **GPU support**: the included demo runs on CPU. A future larger-scale implementation may benefit from CUDA/PyTorch acceleration, but that is not required for the reproducibility example included here.

## Colab and Jupyter Notebook demos

To make NeuSOGA easier to test directly, you can run notebook-based demos in **Google Colab** or locally with **Jupyter Notebook**.

### Required first cell (Meta SAM setup)

NeuSOGA uses Meta's Segment Anything Model (SAM). In Colab, run this block in the **first cell** to install dependencies and download the checkpoint weights:

```bash
!pip install rembg onnxruntime
!pip install opencv-python matplotlib
!pip install git+https://github.com/facebookresearch/segment-anything.git
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

### Google Colab workflow

1. Open your NeuSOGA demo notebook in Colab.
2. Paste and run the required SAM setup block above as the first cell.
3. Run the rest of the notebook cells in order.
4. (Recommended) Enable GPU runtime in Colab: **Runtime → Change runtime type → GPU**.

### Local Jupyter Notebook workflow

1. Start Jupyter from the repository root.
2. Open the NeuSOGA demo notebook.
3. Run the same SAM setup block in the first cell.
4. Execute the remaining cells top-to-bottom.

### Suggested notebook filenames

- `notebooks/NeuSOGA_Demo.ipynb` — end-to-end local demo.
- `notebooks/NeuSOGA_Colab_Demo.ipynb` — Colab-friendly variant.

## Dependency summary

### Runnable demo

The included demo is intentionally lightweight:

- Python 3.10+
- Python standard library only

### Likely external research dependencies for a fuller pipeline

Depending on how the full research codebase evolves, a larger implementation may additionally rely on tools such as:

- PyTorch
- NumPy
- Open3D / point-cloud tooling
- Segment Anything (SAM) and its checkpoint files

These heavier dependencies are **not required** to run the current demo script, but are relevant for notebook workflows that integrate SAM.

## Demo usage

### Quick start: deterministic synthetic example

```bash
python demo/neusoga_demo.py --output-dir outputs/demo
```

### Use a local point cloud file

Supported inputs include simple whitespace- or comma-separated `x y z` text files, plus `.off` files.
For portability, very large input clouds are deterministically downsampled inside the demo before topology construction.

```bash
python demo/neusoga_demo.py \
  --input /path/to/point_cloud.txt \
  --output-dir outputs/demo_from_file
```

### Optionally download ModelNet40

```bash
python demo/neusoga_demo.py \
  --download-modelnet40 \
  --dataset-root data \
  --output-dir outputs/demo_modelnet40
```

### Record a SAM checkpoint path for reproducibility notes

```bash
python demo/neusoga_demo.py \
  --sam-checkpoint /path/to/sam_vit_h_4b8939.pth \
  --output-dir outputs/demo_with_checkpoint_note
```

## Expected outputs

Running the demo writes a compact reproducibility bundle to the chosen output directory:

- `point_cloud.csv` — input or synthetic points with assigned segment IDs,
- `segments.json` — per-segment statistics and primitive hypotheses,
- `symbolic_representation.txt` — human-readable symbolic abstraction,
- `summary.json` — run configuration and reproducibility notes.

These files are intended to make it easy for research readers to inspect how the symbolic representation was produced.

## Practical notes for research users

### ModelNet40

- ModelNet40 is an **external dataset** and is not redistributed here.
- If you use the optional downloader, please ensure that dataset usage complies with the original dataset terms.
- The included demo does not claim full benchmark reproduction from ModelNet40; it only provides a lightweight access path and a symbolic abstraction example.

### SAM checkpoints

- SAM checkpoints are **external artifacts** and must be downloaded manually.
- The included demo records the checkpoint path for provenance, but does not ship checkpoint weights.
- If you extend the pipeline to image-guided segmentation, checkpoint/version provenance should be reported alongside results.

### CPU/GPU behavior

- The included demo is designed to run on **CPU** for portability.
- Selecting `--device cuda` only records intent in the run metadata; it does not enable accelerated kernels in this minimal prototype.
- For larger future experiments, GPU acceleration will likely be preferable for perception and segmentation stages.

## Reproducibility notes

- The synthetic demo is deterministic and does not depend on random initialization.
- Large external assets are intentionally excluded from version control.
- The symbolic outputs are heuristic summaries for demonstration and documentation purposes.
- Readers should treat this repository as a **transparent prototype companion** rather than a final benchmark package.

## Citation

If you use NeuSOGA in academic work, please cite the associated paper once the public arXiv identifier is available.
The BibTeX block below is a **template** and should be updated with the final author list, citation key, and arXiv identifier when the manuscript is public.

```bibtex
@article{li2026neusoga,
  title={Neuro-Symbolic Geometric Abstraction: From Observations to Symbolic Mathematical Representations},
  author={Li, Qingde and others},
  journal={arXiv preprint arXiv:2609.01408},
  year={2026}
  url = {https://arxiv.org/abs/2609.01408}
}
```

## License

This repository is distributed under the MIT License. See `LICENSE`.
