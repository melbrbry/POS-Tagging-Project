# PART OF SPEECH (POS) Tagging
In this project, we use two different LSTM based models for POS Tagging. The first model uses only a simple LSTM network, 
and the we extend this model to a more complex CNN-BI-LSTM network. We demonstrate that LSTM only can achieve high precision rates
on the task in hand. And then we move to CNN-BI-LSTM network where BI-LSTM utilizes both past and future input words, not only the 
past words as it is the case in the simple LSTM, and CNN utilizes the character level features.

## Getting Started

You need to clone the reporistory to your local machine. Run this command in your terminal where you like to clone the project

```
git clone https://github.com/melbrbry/POS-Tagging-Project.git
```

### Prerequisites

Required packages:  
numpy  
keras  
gensim

## Acknowledgement
- This project is done as part of Natural Language Processing course taught by prof. Roberto Navigli - Sapienza Università di Roma.


