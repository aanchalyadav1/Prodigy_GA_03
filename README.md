# Fine-Tuning GPT-2 for Custom Text Generation

This repository contains the code and steps to fine-tune the GPT-2 language model for custom text generation. The fine-tuning process utilizes a dataset derived from the book *Le soleil intérieur* by Adolphe Retté, which is freely available from [Project Gutenberg](https://www.gutenberg.org/ebooks/74828).

## Project Objective
The goal of this project is to train a GPT-2 model to generate coherent and contextually relevant text based on a given prompt. This fine-tuned model will mimic the style and structure of the input training data, creating text that aligns with the book's narrative and linguistic style.

## Dataset
The dataset used for fine-tuning is extracted from *Le soleil intérieur* by Adolphe Retté. The text was preprocessed to remove Project Gutenberg headers/footers and cleaned to enhance training efficiency.

### Dataset Preparation Steps:
1. **Download and Load the Raw Text:**
   - The raw text is loaded from `text.txt`.
2. **Text Cleaning:**
   - Remove Project Gutenberg metadata.
   - Normalize whitespace and remove unwanted characters.
   - Save the cleaned data as `cleaned_text.txt`.
3. **Text Segmentation:**
   - Split the cleaned text into sentences for more structured input.
   - Save the segmented text as `segmented_text.txt`.

## Model Fine-Tuning
The GPT-2 model is fine-tuned using the Hugging Face Transformers library. The steps for fine-tuning are as follows:

### 1. Load Pre-Trained GPT-2
- The pre-trained GPT-2 model and tokenizer are loaded using the `transformers` library.
- A special token (`<|pad|>`) is added to handle padding.

### 2. Tokenize the Dataset
- The segmented text is tokenized and padded to a fixed length of 128 tokens.

### 3. Training Setup
- Training arguments are defined to optimize performance:
  - Batch size: 4 (with gradient accumulation to simulate a batch size of 8)
  - Mixed precision (FP16) enabled for faster training.
  - One epoch for initial testing.

### 4. Training Execution
- The `Trainer` class from the Hugging Face library is used for training.
- The dataset is split into training (90%) and evaluation (10%) subsets.

### Optimizations
To improve training speed and efficiency, the following adjustments were made:
- Lowered learning rate to 3e-5 for stability.
- Reduced batch size and utilized gradient accumulation.
- Enabled mixed precision (FP16).

## Prerequisites
To replicate this project, the following Python packages are required:

```bash
pip install transformers datasets torch tqdm
```

## Files in the Repository
- `text.txt`: The raw text from *Le soleil intérieur*.
- `cleaned_text.txt`: Preprocessed and cleaned text.
- `segmented_text.txt`: Text segmented into sentences.
- `fine_tune_gpt2.py`: Python script for fine-tuning GPT-2.
- `README.md`: This documentation file.

## How to Run
1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. Ensure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the script to start the fine-tuning process:
   ```bash
   python fine_tune_gpt2.py
   ```

## Expected Output
The fine-tuned GPT-2 model will generate text in the style of *Le soleil intérieur*. For example, given a prompt, it will produce coherent sentences that resemble the linguistic and structural patterns of the original text.

## Acknowledgments
- *Le soleil intérieur* by Adolphe Retté: Available from [Project Gutenberg](https://www.gutenberg.org/ebooks/74828).
- Hugging Face Transformers: [https://huggingface.co/transformers/](https://huggingface.co/transformers/).

## License
This project is licensed under the MIT License. Please ensure compliance with Project Gutenberg’s terms when using their content.

---
Feel free to raise issues or contribute to improve the repository!
