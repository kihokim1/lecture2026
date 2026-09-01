# -*- coding: utf-8 -*-
"""이 과목 1~12주차에서 인용한 논문 목록.

주차별로 실제 교재의 「더 읽어보기」에 올라간 것만 넣었다.
서지 실험(run_biblio.py)의 입력이며, 학생이 자기 분야 목록으로 바꿔 쓰면 된다.
"""

# (주차, 짧은 이름, DBLP 검색용 제목, 제1저자 성)
PAPERS = [
    # ── 4주차 프루닝 ────────────────────────────────────────────
    (4, "Han 2015 (연결 학습)", "Learning both Weights and Connections for Efficient Neural Networks", "Han"),
    (4, "Deep Compression", "Deep Compression: Compressing Deep Neural Networks with Pruning, Trained Quantization and Huffman Coding", "Han"),
    (4, "Lottery Ticket", "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks", "Frankle"),
    (4, "Li 2017 (필터 프루닝)", "Pruning Filters for Efficient ConvNets", "Li"),
    # ── 5주차 양자화 ────────────────────────────────────────────
    (5, "Jacob 2018 (정수 추론)", "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", "Jacob"),
    (5, "Krishnamoorthi 백서", "Quantizing deep convolutional networks for efficient inference: A whitepaper", "Krishnamoorthi"),
    (5, "Nagel 백서", "A White Paper on Neural Network Quantization", "Nagel"),
    (5, "Banner 2019 (사후 4비트)", "Post training 4-bit quantization of convolutional networks for rapid-deployment", "Banner"),
    # ── 6주차 지식 증류 ─────────────────────────────────────────
    (6, "Hinton 증류", "Distilling the Knowledge in a Neural Network", "Hinton"),
    (6, "Model Compression", "Model compression", "Bucilua"),
    (6, "Born Again Networks", "Born-Again Neural Networks", "Furlanello"),
    (6, "Beyer 2022 (인내심)", "Knowledge distillation: A good teacher is patient and consistent", "Beyer"),
    (6, "DistilBERT", "DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter", "Sanh"),
    # ── 7주차 NAS ───────────────────────────────────────────────
    (7, "Zoph & Le (NAS)", "Neural Architecture Search with Reinforcement Learning", "Zoph"),
    (7, "MnasNet", "MnasNet: Platform-Aware Neural Architecture Search for Mobile", "Tan"),
    (7, "ShuffleNet V2", "ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design", "Ma"),
    (7, "EfficientNetV2", "EfficientNetV2: Smaller Models and Faster Training", "Tan"),
    (7, "MobileNets", "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications", "Howard"),
    (7, "MobileNetV2", "MobileNetV2: Inverted Residuals and Linear Bottlenecks", "Sandler"),
    (7, "MobileNetV3", "Searching for MobileNetV3", "Howard"),
    # ── 8주차 TinyML ────────────────────────────────────────────
    (8, "MCUNet", "MCUNet: Tiny Deep Learning on IoT Devices", "Lin"),
    (8, "MCUNetV2", "Memory-efficient Patch-based Inference for Tiny Deep Learning", "Lin"),
    (8, "MLPerf Tiny", "MLPerf Tiny Benchmark", "Banbury"),
    (8, "MicroNets", "MicroNets: Neural Network Architectures for Deploying TinyML Applications on Commodity Microcontrollers", "Banbury"),
    (8, "CMSIS-NN", "CMSIS-NN: Efficient Neural Network Kernels for Arm Cortex-M CPUs", "Lai"),
    (8, "Hello Edge", "Hello Edge: Keyword Spotting on Microcontrollers", "Zhang"),
    # ── 9주차 시각 지능 ─────────────────────────────────────────
    (9, "YOLO", "You Only Look Once: Unified, Real-Time Object Detection", "Redmon"),
    (9, "SSD", "SSD: Single Shot MultiBox Detector", "Liu"),
    (9, "Faster R-CNN", "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks", "Ren"),
    (9, "DETR", "End-to-End Object Detection with Transformers", "Carion"),
    (9, "Amdahl 1967", "Validity of the single processor approach to achieving large scale computing capabilities", "Amdahl"),
    # ── 10주차 언어 지능 ────────────────────────────────────────
    (10, "Pope 2023 (스케일링)", "Efficiently Scaling Transformer Inference", "Pope"),
    (10, "AWQ", "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration", "Lin"),
    (10, "LLM.int8()", "LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale", "Dettmers"),
    (10, "GPTQ", "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers", "Frantar"),
    (10, "GQA", "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "Ainslie"),
    (10, "MQA (Shazeer)", "Fast Transformer Decoding: One Write-Head is All You Need", "Shazeer"),
    (10, "vLLM / PagedAttention", "Efficient Memory Management for Large Language Model Serving with PagedAttention", "Kwon"),
    (10, "Orca", "Orca: A Distributed Serving System for Transformer-Based Generative Models", "Yu"),
    (10, "추측 디코딩", "Fast Inference from Transformers via Speculative Decoding", "Leviathan"),
    (10, "추측 표집", "Accelerating Large Language Model Decoding with Speculative Sampling", "Chen"),
    (10, "Roofline", "Roofline: an insightful visual performance model for multicore architectures", "Williams"),
    (10, "Brysbaert 2019", "How many words do we read per minute? A review and meta-analysis of reading rate", "Brysbaert"),
    # ── 11주차 가속기 ───────────────────────────────────────────
    (11, "TVM", "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning", "Chen"),
    (11, "Halide", "Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines", "Ragan-Kelley"),
    (11, "Glow", "Glow: Graph Lowering Compiler Techniques for Neural Networks", "Rotem"),
    (11, "Ansor", "Ansor: Generating High-Performance Tensor Programs for Deep Learning", "Zheng"),
    (11, "Triton", "Triton: an intermediate language and compiler for tiled neural network computations", "Tillet"),
    (11, "DL 컴파일러 서베이", "The Deep Learning Compiler: A Comprehensive Survey", "Li"),
    (11, "Edge TPU 병목", "Google Neural Network Models for Edge Devices: Analyzing and Mitigating Machine Learning Inference Bottlenecks", "Boroumand"),
    (11, "이종 모바일 추론", "Deep Learning Inference on Heterogeneous Mobile Processors: Potentials and Pitfalls", "Liu"),
    # ── 12주차 연합 학습·프라이버시 ─────────────────────────────
    (12, "FedAvg", "Communication-Efficient Learning of Deep Networks from Decentralized Data", "McMahan"),
    (12, "Zhao 비-IID", "Federated Learning with Non-IID Data", "Zhao"),
    (12, "Hsu 디리클레", "Measuring the Effects of Non-Identical Data Distribution for Federated Visual Classification", "Hsu"),
    (12, "Yurochkin (디리클레 최초)", "Bayesian Nonparametric Federated Learning of Neural Networks", "Yurochkin"),
    (12, "FedProx", "Federated Optimization in Heterogeneous Networks", "Li"),
    (12, "SCAFFOLD", "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning", "Karimireddy"),
    (12, "Gboard", "Federated Learning for Mobile Keyboard Prediction", "Hard"),
    (12, "DLG", "Deep Leakage from Gradients", "Zhu"),
    (12, "iDLG", "iDLG: Improved Deep Leakage from Gradients", "Zhao"),
    (12, "Phong (해석적 역산)", "Privacy-Preserving Deep Learning via Additively Homomorphic Encryption", "Phong"),
    (12, "Inverting Gradients", "Inverting Gradients - How easy is it to break privacy in federated learning?", "Geiping"),
    (12, "DP-SGD", "Deep Learning with Differential Privacy", "Abadi"),
    (12, "Secure Aggregation", "Practical Secure Aggregation for Privacy-Preserving Machine Learning", "Bonawitz"),
    (12, "FL 서베이", "Advances and Open Problems in Federated Learning", "Kairouz"),
    (12, "SA 우회", "Eluding Secure Aggregation in Federated Learning via Model Inconsistency", "Pasquini"),
    (12, "Curious Abandon Honesty", "When the Curious Abandon Honesty: Federated Learning Is Not Private", "Boenisch"),
    (12, "Gradient Disaggregation", "Gradient Disaggregation: Breaking Privacy in Federated Learning by Reconstructing the User Participant Matrix", "Lam"),
    (12, "역전 공격 평가", "Evaluating Gradient Inversion Attacks and Defenses in Federated Learning", "Huang"),
    (12, "RAPPOR", "RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response", "Erlingsson"),
    (12, "Apple DP 분석", "Privacy Loss in Apple's Implementation of Differential Privacy on MacOS 10.12", "Tang"),
]
