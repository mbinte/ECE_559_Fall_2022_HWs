# HW05 · Shape classification

[Assignment](assignment.pdf) · [Solution report](solution-report.pdf) · [Full-data results](full-results/)

## Reproduce the full-data run

From the repository root, run:

```sh
cd hw05-shape-classification
python3 -m pip install numpy pillow matplotlib
unzip -q data/geometry-90000.zip -d data
python3 0601-BinteHashem.py
```

This creates the required 8,000/2,000 per-class split and writes the model, metrics, and learning curves to `results/`. The bundled `full-results/` contains the checkpoint, metrics, and plots from the 90,000-image run.

To predict the labels of images in a directory, run the following from that directory:

```sh
python3 /path/to/ECE-559/hw05-shape-classification/0603-BinteHashem.py
```

Inference uses a locally trained checkpoint when available and otherwise uses the bundled full-data checkpoint. Use `--model` to select a specific checkpoint. The course handout specifies student-specific filenames for submissions.
