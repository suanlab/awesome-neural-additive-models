# Awesome Neural Additive Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of papers, implementations, and resources related to **Neural Additive Models (NAMs)** and their variants — interpretable deep learning through additive structures.

Neural Additive Models combine the expressivity of neural networks with the interpretability of Generalized Additive Models (GAMs) by learning a linear combination of neural networks, each attending to a single input feature.

---

## Contents

- [Foundational Work](#foundational-work)
- [Architectural Extensions](#architectural-extensions)
  - [Higher-Order Interactions](#higher-order-interactions)
  - [Sparsity and Feature Selection](#sparsity-and-feature-selection)
  - [Bayesian and Uncertainty-Aware](#bayesian-and-uncertainty-aware)
  - [Distributional Regression (GAMLSS)](#distributional-regression-gamlss)
  - [Scalable Alternatives](#scalable-alternatives)
  - [Other Architectural Variants](#other-architectural-variants)
- [Neural GAM Alternatives](#neural-gam-alternatives)
- [Domain Applications](#domain-applications)
  - [Healthcare and Survival Analysis](#healthcare-and-survival-analysis)
  - [Finance and Credit Scoring](#finance-and-credit-scoring)
  - [Time Series and Forecasting](#time-series-and-forecasting)
  - [Scientific Discovery and Engineering](#scientific-discovery-and-engineering)
  - [Education](#education)
  - [Clustering](#clustering)
  - [Graph Data](#graph-data)
  - [Computer Vision](#computer-vision)
  - [Federated Learning](#federated-learning)
  - [Insurance and Actuarial](#insurance-and-actuarial)
  - [Speech and Language](#speech-and-language)
  - [Medical Imaging](#medical-imaging)
- [Theory and Analysis](#theory-and-analysis)
- [Surveys](#surveys)
- [Open-Source Implementations](#open-source-implementations)
  - [Official and Reference](#official-and-reference)
  - [PyTorch](#pytorch)
  - [R Packages](#r-packages)
  - [Libraries and Toolkits](#libraries-and-toolkits)
  - [NAM Variant Implementations](#nam-variant-implementations)
- [Tutorials and Blog Posts](#tutorials-and-blog-posts)
- [Benchmarks](#benchmarks)
- [Related Non-Neural GAMs](#related-non-neural-gams)

---

## Foundational Work

- **Neural Additive Models: Interpretable Machine Learning with Neural Nets** — Agarwal, Melnick, Frosst, Zhang, Lengerich, Caruana, Hinton — *NeurIPS 2021* — Proposes NAMs: a linear combination of neural networks each attending to a single input feature, combining DNN expressivity with GAM interpretability. [[paper]](https://arxiv.org/abs/2004.13912) [[project]](https://neural-additive-models.github.io/) [[code]](https://github.com/google-research/google-research/tree/master/neural_additive_models)

- **Explainable Neural Networks based on Additive Index Models (xNN)** — Vaughan, Sudjianto, Brahimi, Chen, Nair — *arXiv 2018* — Early precursor proposing structured neural networks based on additive index models for interpretability. [[paper]](https://arxiv.org/abs/1806.01933)

- **Adaptive Explainable Neural Networks (AxNN)** — Chen, Vaughan, Nair, Sudjianto — *arXiv 2020* — Builds structured neural networks from ensembles of GAM networks using boosting or stacking. [[paper]](https://arxiv.org/abs/2004.02353)

---

## Architectural Extensions

### Higher-Order Interactions

- **Higher-Order Neural Additive Models (HONAM)** — Kim, Choi, Kim — *IEEE 2022/2024* — Extends NAMs to capture arbitrary-order feature interactions while maintaining interpretability. [[paper]](https://arxiv.org/abs/2209.15409) [[code]](https://github.com/gim4855744/HONAM)

- **Sparse Interaction Additive Networks (SIAN)** — Enouen, Liu — *NeurIPS 2022* — Extends NAMs from 1D/2D to higher-order (3–5D) sub-networks via tractable interaction detection and sparse selection. [[paper]](https://arxiv.org/abs/2209.09326) [[code]](https://github.com/EnouenJ/sparse-interaction-additive-networks)

- **Neural Additive and Basis Models with Feature Selection and Interactions** — Kishimoto, Yamanishi, Matsuda, Shirakawa — *PAKDD 2024* — Adds a feature selection layer to NAM/NBM enabling two-input networks for interactions in high-dimensional data. [[paper]](https://link.springer.com/chapter/10.1007/978-981-97-2259-4_1)

- **Sparse Deep Additive Model with Interactions (SDAMI)** — Hung, Lin, Calhoun — *arXiv 2025* — Disentangles main effects from interactions using two-stage group lasso sparsity with an "Effect Footprint" concept. [[paper]](https://arxiv.org/abs/2509.23068)

### Sparsity and Feature Selection

- **Sparse Neural Additive Model (SNAM)** — Xu, Bu, Chaudhari, Barnett — *ECML-PKDD 2023* — Employs group LASSO for automatic feature selection within NAMs with provable convergence and exact support recovery. [[paper]](https://arxiv.org/abs/2202.12482)

- **Structural Neural Additive Models** — Luber, Thielmann, Säfken — *arXiv 2023* — Reformulates basis expansion with cubic regression splines as a neural activation layer with learnable knots. [[paper]](https://arxiv.org/abs/2302.09275)

### Bayesian and Uncertainty-Aware

- **EviNAM: Intelligibility and Uncertainty via Evidential Neural Additive Models** — Schleibaum, Thielmann, Teusch, Säfken, Müller — *arXiv 2026* — Integrates evidential learning with NAMs to estimate aleatoric and epistemic uncertainty in a single forward pass. [[paper]](https://arxiv.org/abs/2601.08556)

- **Improving Neural Additive Models with Bayesian Principles (LA-NAM)** — Bouchiat, Immer, Yèche, Rätsch, Fortuin — *ICML 2024* — Augments NAMs with Laplace approximation for credible intervals, implicit feature selection, and interaction ranking. [[paper]](https://arxiv.org/abs/2305.16905) [[code]](https://github.com/fortuinlab/LA-NAM)

- **BayesNAM: Leveraging Inconsistency for Reliable Explanations** — Kim, Park, Choi, Lee, Lee — *arXiv 2024* — Integrates Bayesian neural networks and feature dropout into NAMs for more reliable interpretations. [[paper]](https://arxiv.org/abs/2411.06367)

- **Gaussian Process Neural Additive Models (GP-NAM)** — Zhang, Barr, Paisley — *AAAI 2024* — Replaces per-feature neural networks with random Fourier feature GP approximations, yielding a convex objective. [[paper]](https://arxiv.org/abs/2402.12518) [[code]](https://github.com/Wei2624/GPNAM)

### Distributional Regression (GAMLSS)

- **Neural Additive Models for Location Scale and Shape (NAMLSS)** — Thielmann, Kruse, Kneib, Säfken — *AISTATS 2024* — Extends NAMs to model all distribution parameters (location, scale, shape), enabling prediction intervals. [[paper]](https://arxiv.org/abs/2301.11862) [[code]](https://github.com/AnFreTh/NAMpy)

- **NODE-GAMLSS** — De, Thielmann, Säfken — *NeurIPS 2024 Workshop* — Combines NODE-GAM architecture with GAMLSS for scalable, interpretable distributional regression. [[paper]](https://openreview.net/forum?id=sWvTLlRkzy)

- **NBMLSS: Probabilistic Forecasting via Neural Basis Models for Location Scale and Shape** — Brusaferri, Ramin, Ballarino — *arXiv 2024* — Applies the NBM framework to distributional GAMLSS forecasting of electricity prices. [[paper]](https://arxiv.org/abs/2411.13921)

- **Quantile Neural Basis Models (QNBM)** — Brusaferri, Ramin, Ballarino — *arXiv 2025* — Incorporates quantile GAM principles into neural basis models for electricity price forecasting without distributional assumptions. [[paper]](https://arxiv.org/abs/2509.14113)

### Scalable Alternatives

- **NODE-GAM: Neural Generalized Additive Model for Interpretable Deep Learning** — Chang, Caruana, Goldenberg — *ICLR 2022 (Spotlight)* — Uses differentiable oblivious decision trees as sub-network backbone, enabling self-supervised pre-training. [[paper]](https://arxiv.org/abs/2106.01613) [[code]](https://github.com/zzzace2000/nodegam)

- **Neural Basis Models for Interpretability (NBM)** — Radenovic, Dubey, Mahajan — *NeurIPS 2022* — Shared basis decomposition replacing per-feature networks, achieving orders-of-magnitude parameter reduction. [[paper]](https://arxiv.org/abs/2205.14120) [[code]](https://github.com/facebookresearch/nbm-spam)

- **Scalable Interpretability via Polynomials (SPAM)** — Dubey, Radenovic, Mahajan — *NeurIPS 2022* — Tensor rank decompositions of polynomials to learn scalable GAMs with higher-order interactions. [[paper]](https://arxiv.org/abs/2205.14108) [[code]](https://github.com/facebookresearch/nbm-spam)

### Other Architectural Variants

- **ProtoNAM: Prototypical Neural Additive Models** — Xiong, Sinha, Zhang — *ACM TKDD 2024* — Introduces prototypes into NAM with gradient-boosting-inspired hierarchical shape function modeling. [[paper]](https://arxiv.org/abs/2410.04723) [[code]](https://github.com/teddy-xionggz/protonam)

- **Generalizing Neural Additive Models via Statistical Multimodal Analysis (MNAM)** — Kim, Di Martino, Sapiro — *ICML Workshop 2023* — Mixture of NAMs learning multimodal feature–output relationships with mode probabilities. [[paper]](https://openreview.net/forum?id=xLg8ljlEba) [[code]](https://github.com/youngkyungkim93/MNAM)

- **Neural Additive Experts** — Xiong et al. — *arXiv 2026* — Mixture-of-experts framework with context-gated per-feature experts, relaxing rigid additive constraints. [[paper]](https://arxiv.org/abs/2602.10585)

- **GAMformer: In-Context Learning for Generalized Additive Models** — Mueller, Siems, Nori, Salinas, Zela, Caruana, Hutter — *NeurIPS 2024 Workshop* — First tabular foundation model for GAMs using a transformer to estimate shape functions in a single forward pass. [[paper]](https://arxiv.org/abs/2410.04560)

- **NAMformer: Beyond Black-Box Predictions in Tabular Transformer Networks** — Thielmann, Reuter, Säfken — *arXiv 2025* — Combines tabular transformer with shallow interpretable feature networks to identify marginal feature effects. [[paper]](https://arxiv.org/abs/2504.08712) [[code]](https://github.com/OpenTabular/NAMpy)

- **DNAMite: Interpretable Calibrated Survival Analysis with Discretized Additive Models** — Van Ness, Block, Udell — *arXiv 2024* — Uses feature discretization and kernel smoothing for calibrated, interpretable survival analysis. [[paper]](https://arxiv.org/abs/2411.05923)

- **MT-NAM: An Efficient and Adaptive Model for Epileptic Seizure Detection** — Afzal, Cevher, Shoaran — *arXiv 2025* — Distilled micro tree-based NAM achieving 100× faster inference for EEG seizure detection. [[paper]](https://arxiv.org/abs/2503.08251)

- **M-GAM: Interpretable Generalized Additive Models for Datasets with Missing Values** — McTavish, Donnelly, Seltzer, Rudin — *NeurIPS 2024* — Sparse GAM with missingness indicators and l0 regularization for incomplete data. [[paper]](https://arxiv.org/abs/2412.02646)

- **Interpretable Clinical Classification with Kolmogorov-Arnold Additive Models (KAAM)** — Almodovar, Apellaniz, Garrido, Fernandez-Salvador, Zazo, Parras — *arXiv 2025* — KAN-based additive models delivering transparent symbolic formulas for clinical tabular data. [[paper]](https://arxiv.org/abs/2509.16750)

---

## Neural GAM Alternatives

These papers propose alternative neural network approaches to learning GAMs, distinct from the original NAM architecture.

- **GAMI-Net: Explainable Neural Network based on GAMs with Structured Interactions** — Yang, Zhang, Sudjianto — *Pattern Recognition 2021* — Disentangled feedforward network with sparsity, heredity, and marginal clarity constraints for main effects and pairwise interactions. [[paper]](https://arxiv.org/abs/2003.07132) [[code]](https://github.com/SelfExplainML/GamiNet)

- **Interpretable Generalized Additive Neural Networks (IGANN)** — Kraus, Tschernutter, Weinzierl, Zschech — *EJOR 2024* — Uses gradient boosting with extreme learning machines, reducing training to regularized linear regressions. [[paper]](https://doi.org/10.1016/j.ejor.2023.06.032) [[code]](https://github.com/MathiasKraus/igann)

- **IGANN Sparse: Bridging Sparsity and Interpretability** — Stoecker, Hambauer, Zschech, Kraus — *ECIS 2024* — Extends IGANN with non-linear feature selection producing models with as few as 4% of features. [[paper]](https://arxiv.org/abs/2403.11363) [[code]](https://github.com/MathiasKraus/igann)

- **neuralGAM: Explainable GANNs with Independent Neural Network Training** — Ortega-Fernandez, Sestelo, Villanueva — *Statistics & Computing 2024* — Trains independent neural networks per feature using classical backfitting, ensuring provable additivity. [[paper]](https://doi.org/10.1007/s11222-023-10320-5) [[code (R)]](https://github.com/inesortega/neuralGAM) [[code (Python)]](https://github.com/inesortega/pyNeuralGAM)

- **neuralGAM: An R Package for Fitting Generalized Additive Neural Networks** — Ortega-Fernandez, Sestelo — *arXiv 2025* — R package with flexible architectures and Monte Carlo Dropout uncertainty bands. [[paper]](https://arxiv.org/abs/2505.08610) [[code]](https://github.com/inesortega/neuralGAM)

- **Neural-ANOVA: Analytical Model Decomposition using Automatic Integration** — Limmer, Udluft, Otte — *MLSP 2025* — Decomposes neural networks into GAM-like models using functional ANOVA with closed-form integral evaluation. [[paper]](https://arxiv.org/abs/2408.12319)

- **Tensor Product Neural Networks for Functional ANOVA (ANOVA-TPNN)** — Park, Kong, Choi, Park, Kim — *arXiv 2025* — Guarantees unique functional ANOVA decomposition for stable additive component estimation. [[paper]](https://arxiv.org/abs/2502.15215)

- **Achieving Interpretable ML by Functional Decomposition into Explainable Predictor Effects** — Kohler, Rügamer, Schmid — *arXiv 2024* — Decomposes black-box models into NAM-like main effects and interactions via post-hoc orthogonalization. [[paper]](https://arxiv.org/abs/2407.18650)

---

## Domain Applications

### Healthcare and Survival Analysis

- **Extending NAM for Survival Analysis with EHR Data (TimeNAM)** — Peroni, Kurban, Yang, Kim, Kang, Song — *arXiv 2022* — Extends NAM with pairwise interaction networks and non-proportional Cox losses for gastric cancer prediction. [[paper]](https://arxiv.org/abs/2211.07814)

- **SurvNAM: The Machine Learning Survival Model Explanation** — Utkin, Satyukov, Konstantinov — *Neural Networks 2022* — Adapts NAM as a post-hoc surrogate for black-box survival models via GAM-extended Cox model. [[paper]](https://arxiv.org/abs/2104.08903)

- **CoxNAM: An Interpretable Deep Survival Analysis Model** — Xu, Guo — *Expert Systems with Applications 2023* — Embeds NAMs within Cox proportional hazards for end-to-end nonlinear survival learning. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0957417423007200)

- **CRISP-NAM: Competing Risks Interpretable Survival Prediction with NAMs** — Ramachandram, Raval — *EXPLIMED @ ECAI 2025* — Extends NAM to model cause-specific hazards in competing risks survival analysis. [[paper]](https://arxiv.org/abs/2505.21360)

- **Interpretable Fine-Gray Deep Survival Model for Competing Risks** — Ramachandram, Loefler, Roberts, Verma, Norman, Razak, Pow, de Mestral — *arXiv 2025* — NAM-based Fine-Gray model for competing risks, applied to diabetic foot complications. [[paper]](https://arxiv.org/abs/2511.12409)

- **CoxSENAM: Self-Explaining Neural Networks with Cox Proportional Hazards** — Alabdallah, Hamed, Ohlsson, Rognvaldsson, Pashami — *arXiv 2024* — Combines self-explaining neural networks with NAMs for stable survival analysis explanations. [[paper]](https://arxiv.org/abs/2407.13849)

- **ADHAM: Additive Deep Hazard Analysis Mixtures** — Ketenci, Jeanselme, Reyes Nieva, Joshi, Elhadad — *arXiv 2025* — Interpretable additive survival model with latent subgroup structure for population/individual explanations. [[paper]](https://arxiv.org/abs/2509.07108)

- **Towards Reducing Diagnostic Errors with Interpretable Risk Prediction** — McInerney, Dickinson, Flynn, Young, Young, van de Meent, Wallace — *NAACL 2024* — Combines LLM-extracted EHR evidence with NAMs for interpretable ICU risk prediction. [[paper]](https://arxiv.org/abs/2402.10109)

- **Optimized IGANN for Human Brain Diagnosis using Medical Imaging** — *Knowledge-Based Systems 2024* — IGANN framework for MRI-based brain tumor classification achieving 99.61% accuracy on BRaTS 2021. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0950705124014965)

### Finance and Credit Scoring

- **Monotonic Neural Additive Models for Credit Scoring** — Chen, Ye — *ACM ICAIF 2022* — Enforces monotonicity constraints required by financial regulations while maintaining NAM accuracy. [[paper]](https://arxiv.org/abs/2209.10070)

- **Monotonicity for AI Ethics and Society** — Chen, Ye — *arXiv 2023* — Evaluates monotonic NAMs across criminology, education, healthcare, and finance for AI fairness. [[paper]](https://arxiv.org/abs/2301.07060)

- **SAINTNet: Sparse-Enhanced Additive Interaction Neural Network for Credit Decision** — Lan, Fan, Liu — *Decision Support Systems 2025* — Dual-node additive modules with adaptive sparse feature selection for credit scoring. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0167923625001083)

### Time Series and Forecasting

- **Neural Additive Time-Series Models (NATMs)** — Jo, Kim — *Expert Systems with Applications 2023* — Extends NAM to multivariate time-series prediction with interpretable per-value importance scores. [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0957417423008096)

- **Hierarchical Neural Additive Models for Demand Forecasts (HNAM)** — Feddersen, Cleophas — *Int. Journal of Forecasting 2025* — Hierarchical interaction structure for time-series forecasting, competitive with Temporal Fusion Transformers. [[paper]](https://arxiv.org/abs/2404.04070)

- **Neural Additive Models for Nowcasting** — Kim — *arXiv 2022* — Applies NAM to multivariate nowcasting with explanatory importance for each input. [[paper]](https://arxiv.org/abs/2205.10020)

- **Neural Additive Vector Autoregression Models (NAVAR)** — Bussmann, Nys, Latré — *Discovery Science 2021* — Additive neural networks within VAR framework for nonlinear Granger causal discovery. [[paper]](https://arxiv.org/abs/2010.09429) [[code]](https://github.com/bartbussmann/NAVAR)

- **FocusLearn: Fully-Interpretable, High-Performance Modular Neural Networks for Time Series** — Su, Kloukinas, Garcez — *arXiv 2024* — Modular architecture with attention-based feature selection and independent deep networks per feature; outperforms NAMs on time series tasks. [[paper]](https://arxiv.org/abs/2311.16834)

### Scientific Discovery and Engineering

- **NAM-CAM: Neural-Additive Models for Semi-analytic Descriptions of CAM Simulations** — Ditschuneit, Frenk, Frings, Rudel, Dietzel, Otterbach — *FAIM 2023* — NAMs for computational aeroacoustics/mechanics simulations of turbine blisk design. [[paper]](https://link.springer.com/chapter/10.1007/978-3-031-38241-3_30)

- **Neural Term Structure of Additive Process for Option Pricing** — Lin, Liu — *arXiv 2024* — Neural networks for term structure of additive processes in S&P 500 option pricing. [[paper]](https://arxiv.org/abs/2408.01642)

### Education

- **Explainable Automatic Grading with Neural Additive Models** — Condor, Pardos — *AIED 2024* — NAM for automatic short answer grading with interpretable feature-level explanations. [[paper]](https://arxiv.org/abs/2405.00489)

### Clustering

- **NeurCAM: Interpretable Neural Clustering via Additive Models** — Upadhya, Cohen — *ECAI 2024* — Extends NAMs to unsupervised clustering with fuzzy membership and additive explanations. [[paper]](https://arxiv.org/abs/2408.13361)

### Graph Data

- **Graph Neural Additive Networks (GNAN)** — Bechler-Speicher, Globerson, Gilad-Bachrach — *NeurIPS 2024* — First interpretable-by-design GNN adapting GAMs to graph data with global/local explanations. [[paper]](https://arxiv.org/abs/2406.01317) [[code]](https://github.com/mayabechlerspeicher/Graph-Neural-Additive-Networks---GNAN)

- **Graph Mixing Additive Networks (GMAN)** — Bechler-Speicher, Zerio, Huri, Vestergaard, Gilad-Bachrach, Jess, Bhatt, Sazonovs — *arXiv 2025* — Extends GNANs to sparse time-series as directed graphs with controllable interpretability-expressivity trade-off. [[paper]](https://arxiv.org/abs/2509.23923)

- **Identifying Critical Phases for Disease Onset with Sparse Haematological Biomarkers** — Zerio, Bechler-Speicher, Jess, Sazonovs — *arXiv 2025* — Uses GNANs to model biomarker trajectories as time-weighted directed graphs for disease prediction. [[paper]](https://arxiv.org/abs/2503.14561)

### Computer Vision

- **INAM: Image-Scale Neural Additive Models** — *ESANN 2025* — Adapts NAM to image classification with interpretable per-pixel decision maps. [[paper]](https://www.esann.org/sites/default/files/proceedings/2025/ES2025-54.pdf)

- **Neural Additive Image Model: Interpretation through Interpolation** — Reuter, Thielmann, Säfken — *arXiv 2024* — Combines NAMs with Diffusion Autoencoders to interpret image effects on predictions. [[paper]](https://arxiv.org/abs/2405.02295)

- **Interpretable Similarity of Synthetic Image Utility** — Gatoula, Dimas, Iakovidis — *arXiv 2025* — NAM-inspired interpretable measure for assessing synthetic medical image quality. [[paper]](https://arxiv.org/abs/2512.17080)

### Federated Learning

- **FedNAMs: Performing Interpretability Analysis in Federated Learning Context** — Nanda, Balija, Sahoo — *arXiv 2025* — Deploys NAMs within federated learning for interpretable analysis while preserving data privacy. [[paper]](https://arxiv.org/abs/2506.17466)

- **FedNAM+: The FedNAM+ Conformal Revolution** — Balija, Nanda, Sahoo — *arXiv 2025* — Extends FedNAMs with conformal prediction for uncertainty-quantified federated interpretability. [[paper]](https://arxiv.org/abs/2506.17872)

- **Multi-Level Additive Modeling for Structured Non-IID Federated Learning** — Chen, Zhou, Long, Ma, Jiang, Zhang — *arXiv 2024* — Sums global, subgroup, and client-specific model outputs in an additive structure for non-IID data. [[paper]](https://arxiv.org/abs/2405.16472)

### Insurance and Actuarial

- **An Interpretable Deep Learning Model for General Insurance Pricing (Actuarial NAM)** — Laub, Pho, Wong — *arXiv 2025* — NAM with architectural constraints for sparsity, smoothness, and monotonicity in insurance pricing. [[paper]](https://arxiv.org/abs/2509.08467)

### Speech and Language

- **Speech as a Biomarker for Disease Detection** — Botelho, Abad, Schultz, Trancoso — *arXiv 2024* — Uses NAMs as glass-box classifiers on acoustic/linguistic features for Alzheimer's and Parkinson's detection. [[paper]](https://arxiv.org/abs/2409.10230)

### Medical Imaging

- **LucidAtlas: Uncertainty-Aware, Covariate-Disentangled Atlas Representations** — Jiao, Bhamidi, Qu, Zdanski, Kimbell et al. — *arXiv 2025* — Extends NAMs for uncertainty-aware atlas construction in medical imaging with covariate disentanglement. [[paper]](https://arxiv.org/abs/2502.08445)

---

## Theory and Analysis

- **Provably Explaining Neural Additive Models** — Bassan, Elboher, Ladner, Sahin, Kretinsky, Althoff, Katz — *ICLR 2026* — Shows NAMs admit efficient provably cardinally-minimal explanations using only logarithmic verification queries. [[paper]](https://arxiv.org/abs/2602.17530)

- **Additive Models Explained: A Computational Complexity Approach** — Bassan et al. — *NeurIPS 2025* — Rigorous complexity analysis of generating explanations for various GAM families. [[paper]](https://arxiv.org/abs/2510.21292)

- **How Interpretable and Trustworthy are GAMs?** — Chang, Tan, Lengerich, Goldenberg, Caruana — *KDD 2021* — Compares multiple GAM algorithms and finds equally accurate GAMs can learn contradictory models. [[paper]](https://arxiv.org/abs/2006.06466) [[code]](https://github.com/zzzace2000/GAMs)

- **Challenging the Performance-Interpretability Trade-off** — Kruschel et al. — *BISE 2024* — Benchmarks 7 GAM variants vs 7 black-box models across 20 datasets (68,500 runs); finds EBM matches black-box accuracy. [[paper]](https://arxiv.org/abs/2409.14429)

- **Challenges in Interpretability of Additive Models** — Zhang, Martinelli, John — *XAI-IJCAI 2024 Workshop* — Identifies non-identifiability issues in GAMs/NAMs and argues for interpretability restraint. [[paper]](https://arxiv.org/abs/2504.10169)

- **Reluctant Interaction Inference after Additive Modeling** — Huang, Panigrahi, Yu, Bien — *arXiv 2025* — Selective inference framework for testing interaction effects after fitting sparse additive models with valid p-values. [[paper]](https://arxiv.org/abs/2506.01219)

- **The Most Important Features in GAMs Might Be Groups of Features** — Bosschieter, Franca, Wolk, Wu, Mehta, Dehoney, Kiss, Baker, Zhao, Caruana, Pohl — *Scientific Reports 2026* — Novel group feature importance for GAMs without retraining, validated on medical case studies. [[paper]](https://arxiv.org/abs/2506.19937)

- **Dual Feature-Based and Example-Based Explanation Methods** — Konstantinov, Kozlov, Kirpichenko, Utkin — *arXiv 2024* — Convex-hull-based dual explanation method using NAMs as a tool for example-based explanations. [[paper]](https://arxiv.org/abs/2401.16294)

- **Bayesian Neural Networks for Functional ANOVA Model** — Park, Kim, Lee, Shin, Kong, Kim — *arXiv 2025* — BNN for functional ANOVA decomposition inferring both architecture and parameters for higher-order interactions. [[paper]](https://arxiv.org/abs/2510.00545)

- **CausalKANs: Interpretable Treatment Effect Estimation with Kolmogorov-Arnold Networks** — *arXiv 2025* — Transforms CATE estimators into KAN-based additive models with pruning for closed-form treatment effect formulas. [[paper]](https://arxiv.org/abs/2509.22467)

- **Statistical Inference for Explainable Boosting Machines** — *AISTATS 2026* — Boulevard regularization for EBMs enabling asymptotically normal predictions with minimax-optimal MSE. [[paper]](https://arxiv.org/abs/2601.18857)

---

## Surveys

- **A Comprehensive Survey on Self-Interpretable Neural Networks** — Ji, Sun, Zhang, Wang, Zhuang, Gong, Shen, Qin, Zhu, Xiong — *IEEE TNNLS 2025* — Reviews self-interpretable NNs covering NAM as a key function-based approach. [[paper]](https://arxiv.org/abs/2501.15638) [[code]](https://github.com/yangji721/Awesome-Self-Interpretable-Neural-Network)

- **Comparison of GAMs and Neural Networks in Applications: A Systematic Review** — Doohan, Kook, Burke — *Expert Systems with Applications 2026* — PRISMA review of 143 papers (430 datasets) comparing GAMs and neural networks. [[paper]](https://arxiv.org/abs/2510.24601)

---

## Open-Source Implementations

### Official and Reference

| Name | Framework | Description |
|------|-----------|-------------|
| [google-research/neural_additive_models](https://github.com/google-research/google-research/tree/master/neural_additive_models) | TensorFlow | Official Google Research implementation |
| [nickfrosst/neural_additive_models](https://github.com/nickfrosst/neural_additive_models) | TensorFlow | Standalone fork by co-author Nick Frosst |
| [agarwl/neural_additive_models](https://github.com/agarwl/neural_additive_models) | TensorFlow | Author Rishabh Agarwal's repo |

### PyTorch

| Name | Description |
|------|-------------|
| [AmrMKayid/nam](https://github.com/AmrMKayid/nam) | Full-featured NAM library with config system and notebooks |
| [kherud/neural-additive-models-pt](https://github.com/kherud/neural-additive-models-pt) | Clean PyTorch re-implementation, pip installable (`nam-pt`) |
| [rish-16/nam-pytorch](https://github.com/rish-16/nam-pytorch) | Lightweight unofficial implementation |
| [lemeln/nam](https://github.com/lemeln/nam) | Multi-task NAM by co-author Levi Melnick |

### R Packages

| Name | Description |
|------|-------------|
| [inesortega/neuralGAM](https://github.com/inesortega/neuralGAM) | CRAN package with backfitting-based independent NN training |

### Libraries and Toolkits

| Name | Framework | Description |
|------|-----------|-------------|
| [udellgroup/dnamite](https://github.com/udellgroup/dnamite) | PyTorch | Scikit-learn-compatible NAMs for regression, classification, and survival analysis with built-in feature selection |
| [AnFreTh/NAMpy](https://github.com/AnFreTh/NAMpy) | PyTorch | Supports NAM, NAMLSS, NAMformer, and distributional regression with formula-based specification |
| [interpretml/interpret](https://github.com/interpretml/interpret) | Python/C++ | Microsoft's InterpretML with EBM (tree-based GAM cousin of NAM) |

### NAM Variant Implementations

| Name | Variant | Framework | Description |
|------|---------|-----------|-------------|
| [zzzace2000/nodegam](https://github.com/zzzace2000/nodegam) | NODE-GAM | PyTorch | Differentiable oblivious decision tree backbone, sklearn-compatible |
| [facebookresearch/nbm-spam](https://github.com/facebookresearch/nbm-spam) | NBM + SPAM | PyTorch | Meta's shared basis models (70× fewer params) and polynomial GAMs |
| [fortuinlab/LA-NAM](https://github.com/fortuinlab/LA-NAM) | LA-NAM | PyTorch | Bayesian NAM via Laplace approximation |
| [gim4855744/HONAM](https://github.com/gim4855744/HONAM) | HONAM | PyTorch | Higher-order interaction NAMs |
| [EnouenJ/sparse-interaction-additive-networks](https://github.com/EnouenJ/sparse-interaction-additive-networks) | SIAN | PyTorch | Sparse high-order interaction detection |
| [Wei2624/GPNAM](https://github.com/Wei2624/GPNAM) | GP-NAM | PyTorch | Gaussian Process NAMs |
| [teddy-xionggz/protonam](https://github.com/teddy-xionggz/protonam) | ProtoNAM | PyTorch | Prototypical NAMs |
| [youngkyungkim93/MNAM](https://github.com/youngkyungkim93/MNAM) | MNAM | — | Multimodal NAMs |
| [SelfExplainML/GamiNet](https://github.com/SelfExplainML/GamiNet) | GAMI-Net | PyTorch/TF | Structured interaction networks |
| [MathiasKraus/igann](https://github.com/MathiasKraus/igann) | IGANN | Python | Extreme learning machine-based GANNs |
| [bartbussmann/NAVAR](https://github.com/bartbussmann/NAVAR) | NAVAR | PyTorch | Additive VAR for causal discovery |
| [mayabechlerspeicher/GNAN](https://github.com/mayabechlerspeicher/Graph-Neural-Additive-Networks---GNAN) | GNAN | PyTorch Geometric | Graph Neural Additive Networks |

---

## Tutorials and Blog Posts

- [NAM Project Page](https://neural-additive-models.github.io/) — Official page with slides and links
- [NAM Presentation Slides (Rich Caruana)](https://neural-additive-models.github.io/assets/nam_slides.pdf) — PDF slides
- [Interpretable Deep Learning Models for Tabular Data — Neural GAMs](https://medium.com/@chkchang21/interpretable-deep-learning-models-for-tabular-data-neural-gams-500c6ecc0122) — Blog post covering NAM, NODE-GAM, and related approaches
- [Neural Basis Models for Interpretability (Towards Data Science)](https://towardsdatascience.com/neural-basis-models-for-interpretability-fd04ac958ff2/) — Explains NBM vs NAM

---

## Benchmarks

| Name | Description |
|------|-------------|
| [facebookresearch/nbm-spam](https://github.com/facebookresearch/nbm-spam) | Compares NBM, SPAM against NAM, EBM, XGBoost, DNNs |
| [LeoGrin/tabular-benchmark](https://github.com/LeoGrin/tabular-benchmark) | "Why do tree-based models still outperform deep learning on tabular data?" |
| [dholzmueller/pytabkit](https://github.com/dholzmueller/pytabkit) | ML model benchmarking toolkit for tabular data |
| [InterpretML Benchmarks](https://github.com/interpretml/interpret/blob/master/benchmarks/) | EBM comparison notebooks |

---

## Related Non-Neural GAMs

These are frequently compared to NAMs in the literature:

- **Explainable Boosting Machine (EBM)** — Lou, Caruana, Gehrke, Hooker; Nori et al. — Tree-based cyclic gradient boosting GAM with automatic interaction detection. The primary non-neural baseline. [[paper (GA²M)]](https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf) [[code]](https://github.com/interpretml/interpret)

- **TabNet: Attentive Interpretable Tabular Learning** — Arik, Pfister — *AAAI 2021* — Sequential attention-based feature selection for tabular data; frequently compared to NAMs. [[paper]](https://arxiv.org/abs/1908.07442) [[code]](https://github.com/dreamquark-ai/tabnet)

---

## Contributing

Contributions welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first. If you find a paper or implementation missing from this list, please open a pull request.

---

## License

[![CC0](https://licensebuttons.net/p/zero/1.0/88x31.png)](https://creativecommons.org/publicdomain/zero/1.0/)
