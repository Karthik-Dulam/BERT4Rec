# BERT4Rec in PyTorch

This project is a reimplementation of the BERT4Rec model in PyTorch. The original BERT4Rec was implemented in TensorFlow.

## Introduction

BERT4Rec is a state-of-the-art sequential recommendation model based on the BERT (Bidirectional Encoder Representations from Transformers) architecture. It leverages the power of transformers to model user behavior sequences for recommendation tasks. The model uses a bidirectional self-attention mechanism to capture the contextual information of user-item interactions, making it highly effective for sequential recommendation. In the paper they show that bidirectional contextual embedding improves performance.

For more details, you can refer to the original BERT4Rec paper: [BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformers](https://arxiv.org/abs/1904.06690).

## Installation

To install the required dependencies, run:

```bash
pip install -r requirements.txt