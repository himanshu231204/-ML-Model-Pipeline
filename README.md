
# AutoML Pipeline 

This repository contains an **Automated Machine Learning (AutoML) pipeline** built in Python. The pipeline handles **data preprocessing, feature encoding, model training, evaluation, and prediction** automatically, making it easy to build and test ML models with minimal manual intervention.
## 🎥 Demo Video
[▶️ Watch the Demo](demo/demo.mp4)
## Features

- **Automated Data Preprocessing:** Handles missing values, scaling, and categorical feature encoding (One-Hot Encoding).  
- **Train-Test Split:** Automatically splits data into training and testing sets for reliable model evaluation.  
- **Model Training & Selection:** Supports multiple regression and classification models (Linear Regression, Random Forest, etc.).  
- **Prediction:** Quickly generate predictions on new datasets.  
- **Evaluation Metrics:** Provides metrics like R², accuracy, or RMSE depending on the problem type.  

## Installation

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
pip install -r requirements.txt
````

## Usage

```python
from automl_pipeline import AutoMLPipeline

# Initialize the pipeline
pipeline = AutoMLPipeline(data='your_dataset.csv', target='target_column')

# Run the pipeline
pipeline.run()

# Get predictions
predictions = pipeline.predict(new_data)
```

## Example

```python
# Example: Linear Regression on car price dataset
pipeline = AutoMLPipeline(data='FordCarPrice.csv', target='Price')
pipeline.run()
predictions = pipeline.predict(new_data)
print(predictions)
```

## Repository Structure

```
AutoML-Pipeline/
│
├── data/                 # Your datasets
├── automl_pipeline.py    # Core AutoML pipeline code
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── examples/             # Example scripts
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with improvements.

## License

This project is licensed under the MIT License.


