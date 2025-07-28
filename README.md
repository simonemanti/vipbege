# vipbege

This package provides tools for feature extraction of pulses in the **VIP experiment**.

## Requirements

This project requires the following Python packages:

- numpy
- scipy
- pandas

You can install them with:

```
pip install numpy scipy pandas
```

## Contents

- **`pulse.py`**
  - Contains the `Pulse` class for single-pulse feature extraction.
- **`pulse_batch.py`**
  - Contains the `PulseBatch` class for performing feature extraction on batches of pulses.
  - Includes a `PulseGenerator` for generating pulses within batch processing.
- **`demo.ipynb`**
  - A demonstration notebook showing how to use the provided classes for feature extraction.

## Package Structure

```
vipbege/
│
├── vipbege/
│   ├── pulse.py
│   ├── pulse_batch.py
│   └── ...
├── demo.ipynb
├── README.md
└── ...
```

## Usage

- Use the `Pulse` class from `pulse.py` for extracting features from individual pulses.
- Use the `PulseBatch` class from `pulse_batch.py` for batch processing of multiple pulses.
- Refer to `demo.ipynb` for practical examples and usage demonstrations.


If you are using this software, consider [citing](#citation)!