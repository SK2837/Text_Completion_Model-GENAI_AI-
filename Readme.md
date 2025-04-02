# Text Completion Model (AdiGPT)

## Project Overview
AdiGPT is a text completion model inspired by the architecture of GPT (Generative Pre-trained Transformer). This implementation uses PyTorch to build a transformer-based language model for generating coherent text completions based on input prompts.

## Features
- Custom transformer-based architecture
- Efficient data loading pipeline
- Bidirectional encoding for improved context understanding
- Configurable hyperparameters for model training
- Text generation capabilities with adjustable sampling parameters

## Project Structure
- **DataLoader/**: Contains the data loading and preprocessing modules
  - `dataloader.py`: Handles loading and tokenization of text data
- **model/**: Contains the model architecture
  - `shazamgpt.py`: Implementation of the transformer model
- **trainer.py/**: Contains training utilities
  - `trainer.py`: Training loop implementation and optimization logic
- **main.py**: Main script to train the model
- **mainrun.py**: Helper script for training
- **run.py**: Script for running inference with the trained model
- **bashrun.sh**: Shell script for simplified model execution
- **bigram.ipynb**: Notebook demonstrating a simple bigram language model
- **torch-examples.ipynb**: Examples of PyTorch usage in the context of this project

## Getting Started

### Prerequisites
```
python 3.9+
pytorch 2.0+
numpy
matplotlib
```

### Installation
1. Clone the repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Training a Model
To train the model with default parameters:
```
python main.py
```

To customize training:
```
python main.py --batch-size 64 --learning-rate 3e-4 --epochs 10
```

### Generating Text
Once trained, you can generate text using:
```
python run.py --prompt "Your prompt text here" --max-tokens 100
```

## Model Architecture
The ShazamGPT model is based on the transformer architecture with:
- Multi-head self-attention mechanism
- Position-wise feed-forward networks
- Layer normalization
- Residual connections
- Learned positional embeddings

## Dataset
The model can be trained on any text dataset. Example datasets included:
- `Stock_Exchange.txt`: Text data related to stock market

## Performance
The model achieves competitive performance on text completion tasks with relatively small computational requirements compared to larger commercial models.

## Future Improvements
- Implement more efficient attention mechanisms
- Support for larger context windows
- Integrate with web interfaces for interactive use
- Fine-tuning capabilities on domain-specific data

## Contact
Created by [Sai Adarsh Kasula] - feel free to contact me for any questions or collaborations!

## License
This project is open source and available under the [MIT License].
