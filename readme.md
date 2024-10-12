# Few-Shot Query Classifier and QnA System

This project builds a QnA system using MS MARCO dataset and fine-tunes a pre-trained model for answering questions. It also includes a few-shot query classifier for categorizing queries.

## Project Structure
- `data/`: Contains the dataset files.
- `models/`: Scripts for loading and fine-tuning the QA model and query classifier.
- `app/`: The Streamlit-based UI for user interaction.
- `utils/`: Utility functions for evaluation.

## Running the Project
1. Install dependencies: `pip install -r requirements.txt`
2. Fine-tune the QA model: Run the script in `models/qa_model.py`
3. Launch the Streamlit app: `streamlit run app/qa_app.py`

## Dataset
The project uses a small subset of the MS MARCO dataset for training and evaluation.