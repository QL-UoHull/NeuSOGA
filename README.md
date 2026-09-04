# NeuSOGA

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](: Research Prototype](https://img.shields.io/badge/status-research%20prototypeL-UoHull/NeuSOGA)
[![arXiv](https://img.shields.io/badge/arXiv-2609.01408-B31B1B.svg?logo=arxiv)](https://arxiv09.01408)

# Neuro-Symbolic Geometric Abstraction (NeuSOGA)

### From Observations to Symbolic Mathematical Representations

NeuSOGA is a neuro-symbolic framework for transforming observations into explicit symbolic mathematical representations through topology-guided geometric abstraction.

The framework implements the abstraction hierarchy:

```text
Observation (O)
        ↓
Topology (T)
        ↓
Geometry (G)
        ↓
Symbol (S)
```

where observations are progressively transformed into topological abstractions, geometric abstractions, and ultimately symbolic mathematical representations.

<p align="center"> <img src="assets/figures/Neuro-symbolic-pipeline.jpg" widthcenter"> <em>Overview of the NeuSOGA framework.</em> </p>


---

# Research Motivation

Modern AI systems often encode knowledge within latent neural representations. While highly effective for perception and prediction, these representations are typically difficult to inspect, manipulate, or analyse mathematically.

NeuSOGA explores an alternative paradigm:

```text
Observation
      ↓
Topology
      ↓
Geometry
      ↓
Symbol
```

Rather than learning geometry solely through statistical optimization, NeuSOGA extracts topology-aware structure, performs adaptive geometric abstraction, and synthesizes explicit symbolic mathematical representations using analytical geometry.

The central research question is:

> How can symbolic mathematical representations emerge from observations through a neuro-symbolic abstraction process?

---

# NeuSOGA Pipeline

NeuSOGA follows a four-stage abstraction hierarchy.

## 1. Observation (O)

Geometric observations are acquired from point clouds or projected views.

Current implementation supports:

- ModelNet40 point-cloud objects
- Arbitrary-view projections
- Normalized geometric observations

The observation layer provides the raw spatial data from which symbolic representations are ultimately constructed.

---

## 2. Topology (T)

Intrinsic structure is extracted through topology-guided analysis.

NeuSOGA computes:

- Euclidean Distance Transforms (EDT)
- Topology-aware structural cores
- Distance-field maxima
- Internal void structures
- Global spatial organization

The resulting topology nodes provide compact abstractions of object structure and subsequently guide perception.

---

## 3. Geometry (G)

Topology-aware structure is transformed into sparse geometric abstractions.

The geometry stage combines:

- Topology-guided Segment Anything (SAM) prompting
- Contour extraction
- Adaptive multi-scale scale-space abstraction
- Hybrid coarse-to-fine boundary refinement
- Sparse control-polygon generation

Dense observations are compressed into compact and interpretable geometric representations while preserving important structural detail.

---

## 4. Symbol (S)

The symbolic stage converts geometric abstractions into explicit mathematical representations.

NeuSOGA employs Implicit Area Splines to generate:

- Analytical implicit representations
- Arbitrary-order smoothness
- Additive composition
- Closed-form evaluation
- Symbolic geometric models

The final representation is expressed as:

```text
F(x,y) = 0
```

providing an explicit symbolic mathematical description of the observed shape.

---

# Repository Contents

```text
NeuSOGA/
│
├── assets/
│   └── figures/
│
├── demo/
│   └── neusoga_demo.py
│
├── notebooks/
│
├── paper/
│
├── README.md
├── requirements-demo.txt
├── LICENSE
└── CITATION.cff
```

This repository contains the reference implementation accompanying the NeuSOGA paper.

---

# Installation and Setup

NeuSOGA was developed and validated primarily in **Google Colab**. The easiest way to reproduce the results is through Colab or a compatible Jupyter environment.

---

## Google Colab Setup

Run the following commands in the first notebook cell:

```bash
!pip install rembg
!pip install opencv-python matplotlib
!pip install git+https://github.com/facebookresearch/segment-anything.git
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

The downloaded checkpoint:

```text
sam_vit_b_01ec64.pth
```

will be used automatically by the NeuSOGA pipeline.

### Recommended Colab Configuration

```text
Runtime → Change runtime type → GPU
```

GPU acceleration is recommended but not required.

---

## Local Python Environment

### 1. Clone the Repository

```bash
git clone https://github.com/QL-UoHull/NeuSOGA.git

cd NeuSOGA
```

### 2. Create a Virtual Environment

#### Linux / macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

#### Windows

```cmd
python -m venv .venv

.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install rembg
pip install opencv-python matplotlib
pip install git+https://github.com/facebookresearch/segment-anything.git
```

### 4. Download SAM Checkpoint

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

or manually download from:

https://github.com/facebookresearch/segment-anything

---

# External Resources

## ModelNet40 Dataset

NeuSOGA automatically downloads and extracts the ModelNet40 dataset when executed for the first time.

The dataset is not bundled with this repository and no manual download is required.

---

## Segment Anything (SAM)

NeuSOGA employs topology-guided perception using Meta's Segment Anything Model.

Required checkpoint:

```text
sam_vit_b_01ec64.pth
```

Download:

```bash
wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

---

## Hardware Requirements

The framework automatically detects:

```text
CPU
CUDA GPU
```

CPU execution is fully supported.

GPU acceleration is recommended for large-scale robustness experiments but is not required.

---

# Running NeuSOGA

## Google Colab

After executing the installation cell:

```python
!python neusoga_demo.py
```

or execute the notebook cells sequentially.

---

## Local Execution

Run:

```bash
python neusoga_demo.py
```

The script automatically:

1. Downloads ModelNet40 (if required).
2. Loads SAM.
3. Processes representative objects from all 40 ModelNet40 categories.
4. Generates arbitrary-view projections along:

```text
[1, 1, 1]
```

5. Executes the complete:

```text
O → T → G → S
```

abstraction hierarchy.

---

# Outputs

Results are written to:

```text
robustness_results/
```

For each object, NeuSOGA generates an eight-stage visualization illustrating:

```text
1. Observation (O)
2. Euclidean Distance Transform
3. Topology Nodes (T)
4. Topology-Guided Segmentation
5. Scale-Space Contour
6. Control Polygon (G)
7. Area Spline Field
8. Symbolic Boundary F(x,y)=0 (S)
```

These visualizations provide a transparent view of how symbolic mathematical representations emerge from geometric observations.

---

# Colab and Jupyter Notebook Workflows

## Google Colab

Recommended workflow:

1. Open the notebook in Colab.
2. Run the dependency installation cell shown above.
3. Ensure the SAM checkpoint has been downloaded.
4. Enable GPU runtime (optional but recommended).
5. Execute all notebook cells sequentially.

---

## Local Jupyter Notebook

Launch Jupyter:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Then:

1. Open the desired NeuSOGA notebook.
2. Install the required dependencies.
3. Download the SAM checkpoint.
4. Execute notebook cells in order.

---

## Suggested Notebooks

```text
notebooks/NeuSOGA_Demo.ipynb
```

End-to-end demonstration of the NeuSOGA pipeline.

```text
notebooks/NeuSOGA_Robustness.ipynb
```

Robustness evaluation across object categories and viewpoints.

---

# Robustness Evaluation

The current implementation evaluates NeuSOGA across:

- 40 object categories from ModelNet40
- Arbitrary-view projections
- Topology-aware segmentation
- Adaptive geometric abstraction
- Symbolic mathematical representation formation

The resulting visualizations demonstrate that the abstraction hierarchy remains stable across diverse geometric structures and viewing conditions.

---

# Explainability by Construction

A central objective of NeuSOGA is explainability through explicit representation.

Unlike conventional latent neural representations, every stage of the abstraction hierarchy remains observable:

```text
Observation
      ↓
Topology
      ↓
Geometry
      ↓
Symbol
```

The generated symbolic models can be inspected, edited, analysed, and evaluated directly through their mathematical representation.

---

# Paper

**Neuro-Symbolic Geometric Abstraction (NeuSOGA): From Observations to Symbolic Mathematical Representations**

arXiv:

https://arxiv.org/abs/2609.01408

---

# Citation

```bibtex
@article{li2026neusoga,
  title={Neuro-Symbolic Geometric Abstraction (NeuSOGA):
         From Observations to Symbolic Mathematical Representations},
  author={Li, Qingde; et al},
  journal={arXiv preprint arXiv:2609.01408},
  year={2026},
  url={https://arxiv.org/abs/2609.01408}
}
```

---

# License

This repository is distributed under the MIT License.

See `LICENSE` for details.
