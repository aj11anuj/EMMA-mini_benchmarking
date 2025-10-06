# Benchmarking Gemma-3 Models on EMMA-mini Dataset

### Based on the paper: 
**“Can MLLMs Reason in Multimodality? EMMA: An Enhanced MultiModal ReAsoning Benchmark”** </br>
[arXiv:2501.05444](https://arxiv.org/abs/2501.05444)

**Abstract:**
The ability to organically reason over and with both text and images is a pillar of human intelligence, yet the ability of Multimodal Large Language Models (MLLMs) to perform such multimodal reasoning remains under-explored. Existing benchmarks often emphasize text-dominant reasoning or rely on shallow visual cues, failing to adequately assess integrated visual and textual reasoning. We introduce EMMA (Enhanced MultiModal reAsoning), a benchmark targeting organic multimodal reasoning across mathematics, physics, chemistry, and coding. EMMA tasks demand advanced cross-modal reasoning that cannot be addressed by reasoning independently in each modality, offering an enhanced test suite for MLLMs' reasoning capabilities. Our evaluation of state-of-the-art MLLMs on EMMA reveals significant limitations in handling complex multimodal and multi-step reasoning tasks, even with advanced techniques like Chain-of-Thought prompting and test-time compute scaling underperforming. These findings underscore the need for improved multimodal architectures and training paradigms to close the gap between human and model reasoning in multimodality.

### Overview

This repository contains an independent implementation and extension of the EMMA benchmark. Using the **EMMA-mini dataset**, which contains 400 questions (100 each from Physics, Chemistry, Maths, and Coding), this project benchmarks the **Gemma-3 model series** (4B, 12B, 27B) under the same environment and hyperparameter settings described in the original paper.

For detailed explanations and results, refer to the presentation and recording (if available) included in this repository.

### Usage

To run the benchmark:
1. Set your AI Studio API key:
   ```bash
   export AI_STUDIO_API_KEY="your_api_key_here"
   ```
2. Run the benchmark script:
   ```bash
   python gemma.py
   ```

### Attribution

If you use this repository or any part of it in your work, **please provide attribution**. This repository is licensed under the **BSD 3-Clause License** ©2025 Anuj Tiwari. You may use, modify, and distribute this code with proper attribution. See the [LICENSE](./LICENSE) file for details.
