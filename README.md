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

Possible improvements could include:  

- Limiting vocabulary size and adding an `<OOV>` token  
- Using longer sequences across the text  
- Adding stronger regularisation (dropout, early stopping, LR scheduling)  
- Tracking **top-k accuracy** and **perplexity** for evaluation

# RNN Input-Output Architectures  

RNNs can be designed to handle different input–output structures depending on the task.  

## 1. One-to-One  
- **What it is:** A single input produces a single output  
- **Use case:** Traditional neural networks also fall here  
- **Real-life example:** Image classification (Input: an image -> Output: class label)  


## 2. One-to-Many  
- **What it is:** A single input produces a sequence of outputs  
- **Real-life examples:**  
  - Music generation (Input: start note -> Output: full melody)  
  - Text generation (Input: prompt word -> Output: sentence continuation)  

## 3. Many-to-One  
- **What it is:** A sequence of inputs produces a single output  
- **Real-life examples:**  
  - Sentiment analysis (Input: sequence of words in a review -> Output: Positive/Negative)  
  - Fraud detection (Input: sequence of transactions -> Output: Fraud/Not Fraud)  

## 4. Many-to-Many  
- **What it is:** A sequence of inputs produces a sequence of outputs  
- **Two types:**  
  - **Synchronized:** Input and output sequences are the same length  
  - **Unsynchronized:** Input and output sequences differ in length  

**Real-life examples:**  
- **Synchronized:**  
  - Named Entity Recognition (NER): Input = sentence -> Output = tag for each word  
  - POS tagging: Input = words -> Output = parts of speech  

- **Unsynchronized:**  
  - Translation: Input = English sentence -> Output = French sentence  
  - Chatbots: Input = user query -> Output = response sequence  


## Hamlet Project Setup  
- **Type:** Many-to-One RNN  
- **Input:** sequence of 13 words  
- **Output:** the next predicted word  


Streamlit App: https://shakespeare-hamlet-next-word-prediction-with-lstm-g58rtb24dagk.streamlit.app
