# Automatic War Prediction System

This project provides a fully automated pipeline that generates synthetic war prediction data and trains simple machine learning models. When access to online data sources is unavailable, the data loader simulates realistic values based on historical patterns.

## Disclaimer

The datasets produced by this project are synthetic and should not be interpreted as factual predictions. The models are intended for demonstration purposes only.

## Requirements

See `Requirement.txt` for the list of Python packages needed.

## Usage

```bash
python AutomaticWarPredictionSystem.py
```

The script will save reports and visualizations in a newly created results directory.

## Testing

Run the test suite with `pytest -q`. All packages listed in `Requirement.txt` must be installed beforehand.
