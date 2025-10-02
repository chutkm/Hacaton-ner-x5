
# Named Entity Recognition (NER) Service for Product Search Queries!

This repository contains a **Named Entity Recognition (NER)** service designed for extracting entities (`TYPE`, `BRAND`, `VOLUME`, `PERCENT`) from product search queries. The system uses a transformer-based model with an optional CRF layer for sequence labeling, implemented in **PyTorch**, and provides a **FastAPI**-based API for inference.

## Features

- **NER Model**: Utilizes transformer models (e.g., `bert-base-multilingual-cased`) with a modular classifier head and optional CRF layer for token classification.
- **Training Pipeline**: Supports two-stage training:
  - **Stage 1**: Head-only training with frozen transformer base.
  - **Stage 2**: Full fine-tuning with unfrozen base.
- **Hyperparameter Optimization**: Optional **Optuna**-based hyperparameter search.
- **Inference API**: **FastAPI** service for predicting entity spans in text queries.
- **Metrics**: Implements macro F1 score for entity-level strict matching.
- **Modular Model Saving/Loading**: Supports saving and loading individual model components (`BERT`, `classifier`, `CRF`).

## Project Structure

- **`src/`**: Core source code
  - `dataset.py`: Data loading and preprocessing for training and validation datasets.
  - `train.py`: Main training script with support for two-stage training and hyperparameter optimization.
  - `metrics.py`: Entity-level macro F1 score calculation and annotation parsing.
  - `utils.py`: Utility functions for label management, BIO fixing, and model saving/loading.
  - `inference.py`: Inference logic for predicting entity spans from text.
  - `model/modular.py`: Modular token classifier implementation (not included in the provided files but assumed to exist).
- **`api/`**: FastAPI application
  - `app.py`: FastAPI service for serving NER predictions.
  - `log_middleware.py`: Logging middleware for API requests (not included but referenced).
- **`data/`**: Directory for training and validation CSV files (not included).
- **`models/`**: Directory for saving trained models and their components.

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-repo/ner-service.git
   cd ner-service
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure you have **Python 3.8+** installed. Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

   Example `requirements.txt`:
   ```plaintext
   torch>=1.10.0
   transformers>=4.20.0
   datasets>=2.0.0
   pandas>=1.3.0
   numpy>=1.20.0
   optuna>=2.10.0
   fastapi>=0.78.0
   uvicorn>=0.18.0
   ```

4. **Prepare Data**:
   Place your training and validation CSV files in the `data/` directory. The CSV files should have `sample` (text) and `annotation` (list of tuples like `[(start, end, label), ...]`) columns.

## Usage

### Training the Model

Run the training script with default or custom parameters:

```bash
python src/train.py \
    --model_name_or_path bert-base-multilingual-cased \
    --train_csv data/train.csv \
    --val_csv data/val.csv \
    --output_dir models/v1 \
    --max_length 128 \
    --use_crf \
    --stage1_head_only \
    --stage2_full \
    --optuna_trials 10
```

- Use `--optuna_trials 0` to disable hyperparameter search.
- Use `--no-use_crf` to disable the CRF layer.
- Use `--no-stage1_head_only` or `--no-stage2_full` to skip respective training stages.

The trained model and metadata will be saved in `models/v1/final_model/`, with optional run-specific data in `runs/`.

### Running the API

Start the **FastAPI** server:

```bash
MODEL_DIR=models/v1/final_model uvicorn api.app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

#### API Endpoints

- **POST /api/predict**:
  - **Input**: JSON with `input` field (e.g., `{"input": "Coca-Cola 1L"}`)
  - **Output**: List of spans with `start_index`, `end_index`, and `entity` (e.g., `[{"start_index": 0, "end_index": 9, "entity": "B-BRAND"}, {"start_index": 10, "end_index": 12, "entity": "B-VOLUME"}]`)
  - Example:
    ```bash
    curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"input": "Coca-Cola 1L"}'
    ```

- **GET /health**:
  - Returns `{"ok": true}` if the service is running.

- **GET /version**:
  - Returns the model directory path.

### Inference Example

To predict spans programmatically:

```python
from src.inference import predict_spans

text = "Coca-Cola 1L"
spans = predict_spans(text)
print(spans)  # Example: [(0, 9, 'B-BRAND'), (10, 12, 'B-VOLUME')]
```

## Data Format

- **CSV Files**:
  - Columns: `sample` (string), `annotation` (string representation of a list of tuples, e.g., `[(0, 9, 'B-BRAND'), (10, 12, 'B-VOLUME')]`)
  - Separator: `;`

- **Labels**:
  - Supported entities: `TYPE`, `BRAND`, `VOLUME`, `PERCENT`
  - BIO tagging scheme: `O`, `B-TYPE`, `I-TYPE`, `B-BRAND`, `I-BRAND`, etc.

## Notes

- The model supports **multilingual input** via `bert-base-multilingual-cased` or other transformer models (e.g., `ai-forever/ruRoberta-large`, `DeepPavlov/bert-base-bg-cs-pl-ru-cased`).
- The `fix_bio_labels` and `fix_bio_spans` functions ensure consistent BIO tagging during training and inference.
- The system handles edge cases like empty inputs or invalid annotations.
- For production, set the `MODEL_DIR` environment variable to the trained model path.

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/xyz`).
3. Commit your changes (`git commit -am 'Add feature XYZ'`).
4. Push to the branch (`git push origin feature/xyz`).
5. Create a pull request.

## License

This project is licensed under the **MIT License**.
```
