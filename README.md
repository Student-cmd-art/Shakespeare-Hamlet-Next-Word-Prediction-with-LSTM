# Next-Word Prediction Using LSTM (Shakespeare Hamlet)

This project builds a language model on Shakespeare’s *Hamlet* using an LSTM-based Recurrent Neural Network (RNN). The model is trained to predict the next word in a sequence, given a sequence of preceding words.  

## Overview  

- **Task:** Predict the next word in a sequence of text from Shakespeare’s *Hamlet*  
- **Dataset:** NLTK Gutenberg corpus – *shakespeare-hamlet.txt*  
- **Architecture:** Embedding -> LSTM -> Dropout -> LSTM -> Dense (Softmax)  

## Input Handling  

- Text is lowercased and tokenised using Keras `Tokenizer`  
- Vocabulary size = **4,818 unique tokens**  
- Sequences are generated using incremental **N-grams** from each line of the play  
- Maximum sequence length = **14 tokens** (13 input, 1 target)  
- Sequences padded to uniform shape  
- Targets one-hot encoded across the full vocabulary  


## Model Architecture  

| Layer         | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| **Embedding** | Maps token IDs to 100-dim dense vectors                                     |
| **LSTM(150)** | Processes sequence, returns hidden states at each step                      |
| **Dropout**   | Dropout(0.5) to reduce overfitting                                          |
| **LSTM(100)** | Processes sequence, outputs final hidden state                              |
| **Dense**     | 4,818 units with Softmax activation -> probability distribution over vocab   |


## Training Setup  

- **Loss Function:** Categorical Cross-Entropy  
- **Optimizer:** Adam  
- **Batch Size:** 16  
- **Epochs:** 100  
- **Callbacks:** None used (no EarlyStopping)  

## Performance  

During training, I observed:  

- Training accuracy rose to **~0.63**  
- Validation accuracy stagnated around **~0.05**  
- Validation loss increased sharply (**~13.6 by the final epoch**)  

**Interpretation:**  
The model clearly overfitted the small dataset. It memorised patterns in the training set but failed to generalise to unseen sequences.  

## Recommendations  

If I were to extend the project, possible improvements could include:  

- Limiting vocabulary size and adding an `<OOV>` token  
- Using longer sliding window sequences across the text  
- Adding stronger regularisation (dropout, early stopping, LR scheduling)  
- Tracking **top-k accuracy** and **perplexity** for evaluation  
