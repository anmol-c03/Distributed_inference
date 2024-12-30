


# Table of Contents
1. [Project Overview](#project-overview)
2. [Working](#Working)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Notes](#notes)
6. [Future_Improvements](#Future_Improvements)

# Project Overview
This project addresses the challenge of inferencing large-scale AI models, such as modern Large Language Models (LLMs) with billions of parameters (e.g., LLaMA 3.2 with 405B). Due to memory and computational constraints, downloading and running these models locally is impractical.To solve this, the project implements model sharding, where the model is partitioned into smaller chunks and distributed across multiple devices. Each shard is loaded and executed based on the device's specifications and computational capabilities. This prototype demonstrates how distributed inference can make running massive models feasible, showcasing a solution for optimizing deep learning infrastructure.



# Working

It implements Layer-wise Model Initialization.Each layer's weights from a pretrained GPT-2 model are loaded into separate model instances (GPT).The first layer includes embeddings, and the last layer includes the language model head for text generation.Input tokens are processed sequentially through each model. The output logits from each layer are passed as input to the next model.The program uses a top-k sampling approach to generate text autoregressively. A predefined sequence (e.g., "Hello, you doing fine") is used as input, and additional tokens are generated.


## Installation

To use this project, follow these steps:

1. Clone the repository:
```bash
git clone https://github.com/anmol-c03/Distributed_inference.git
```

2. Navigate to the project directory:
```bash
cd Distributed_inference
```


3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

# Notes

- No Real Networking
- Dummy Setup
- Pretrained Weights

# Future_Improvements

- Introduce actual networking for distributed model execution.
- Optimize memory usage for large-scale models.
- Implement a more realistic distributed system using RPC or model-parallel strategies.
