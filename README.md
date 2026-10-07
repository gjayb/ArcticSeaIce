# Manuscript Figures

**Scripts and data to generate figures for the manuscript, "Developing a physics-sensitive reduced-order model to investigate sea ice tipping pathways".** 

## Directory Structure
```text
Figures/
├── README.md              # Documentation
├── requirements.txt       # Python package requirements
├── Scripts/               # Scripts to generate manuscript figures
├── Data/                  # Model output and supporting data used by the scripts
└── Pngs/                  # Figures in png format
```

## Installation

This requires python version `3.13`. It may work in other versions but has only been tested with 3.13.
Install using `uv`.

```bash
uv venv
uv pip install -r requirements.txt
```

Note this assumes that the ArcticIceModel repository is in a separate folder next to this one.
```text
Figures/                    # This directory
ArcticIceModel/             # Model source code
```

## Running

To run each figure script:
```bash
uv run Scripts/figure01_Arctic_map.py
```

The figures are saved to the directory `Figures/Pngs`.

