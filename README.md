# Energy-Efficient ML: Reproducibility Study

**[Re] Reproducibility study of "Optimizing Deep Learning Inference on Embedded Systems Through Adaptive Model Selection"**

This repository contains the code and report for our reproducibility study of the paper *"Optimizing Deep Learning Inference on Embedded Systems Through Adaptive Model Selection"* by V.S. Marco et al. Our study focuses on replicating the methods and results in the domains of image classification and machine translation.

**For a comprehensive overview of our findings and methodology, please refer to our detailed report: [article.pdf](./article.pdf).**

## Scope of Reproducibility

In this study, we aim to reproduce the findings of Marco et al. by evaluating:
- **Claim 1:** Improvement in top-1 accuracy for image classification by using a premodel selection pipeline.
- **Claim 2:** Reduction in inference time (1.8x) for image classification compared to using the most accurate model (ResNet152).
- **Claim 3:** Reduction in machine translation inference time (1.34x) without significant loss in accuracy.

## Methodology

We implemented the majority of the code from scratch, focusing on both image classification and machine translation. Our work includes feature extraction, premodel construction, and adaptive model selection.

## Results

Our results diverged significantly from those reported in the original paper, particularly due to the lack of specific methodological details. While we achieved similar performance in the image classification premodel, our machine translation results did not meet the expectations set by the original authors.

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

You can adjust details based on the specific content of your project.