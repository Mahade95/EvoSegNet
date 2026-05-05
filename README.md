# EvoSegNet: Supplementary Material
This repository contains supplementary experimental results and architectural analysis for:

**EvoSegNet: Multi-Scale and Uncertainty-Aware Evolutionary Neural Architecture Search for Efficient 3D Medical Image Segmentation**

---



## 1. BraTS 2021 – Architectural Search Analysis

### DSC vs FLOPs vs Filters

<p align="center">
  <img src="BraTs__DSC_vs_Flops_vs_Filter-01.jpg" width="70%">
</p>

### DSC vs FLOPs vs Parameters

<p align="center">
  <img src="BraTs__DSC_vs_Flops_vs_Params-01.jpg" width="70%">
</p>
---

## 2. ACDC – Architectural Search Analysis

### DSC vs FLOPs vs Filters

<p align="center">
  <img src="ACDC_DSC_vs_Flops_vs_Filter-01.jpg" width="70%">
</p>

### DSC vs FLOPs vs Parameters

<p align="center">
  <img src="ACDC_DSC_vs_Flops_vs_Params-01.jpg" width="70%">
</p>

---

## Reproducibility

All experiments were conducted under the following configuration:

- Framework: TensorFlow
- GPU Acceleration: CUDA + cuDNN
- Hardware: NVIDIA RTX 3090
- Fixed random seeds for deterministic behaviour
- Identical dataset splits and search settings as described in the manuscript

---

## Description

The figures above illustrate:

- Multi-objective Pareto search behaviour (DSC vs computational complexity)
- Trade-offs between segmentation accuracy, parameter count, and FLOPs
- Convergence dynamics in terms of DSC, HD95, and IoU
- Stability of the evolutionary optimisation process

These supplementary results provide further evidence of EvoSegNet’s efficiency–accuracy balance across both brain tumour (BraTS 2021) and cardiac (ACDC) segmentation benchmarks.


<h2>Independent Run Consistency Analysis</h2>

This section presents the independent run consistency and variance analysis of EvoSegNet on the BraTS 2021 and ACDC datasets. The model was trained across three independent runs, each for 100 epochs. To evaluate stability, per-epoch statistics were computed as mean $\pm$ standard deviation (SD) for both training and validation phases. The evaluation metrics are organised based on clinical relevance, including dice similarity coefficient (DSC, higher is better), intersection over union (IoU, higher is better), and the 95th percentile Hausdorff Distance (HD95, in millimetres, lower is better).

The epoch-wise trajectories for each independent run are presented in below Figure, where each figure jointly illustrates the results for the BraTS 2021 and ACDC datasets. Across all runs, both datasets demonstrate smooth convergence behaviour with minimal variation between training and validation curves. The low dispersion in DSC and IoU, together with consistently decreasing HD95, indicates stable optimisation and reliable boundary delineation across independent trainings.


<hr>

<h3>🧠 BraTS 2021 – Independent Runs</h3>

<h4>🔹🧠 BraTS 2021 – Run 1</h4>
<p align="center">
  <img src="BraTS_Run_1_metrics-01.jpg" width="70%">
</p>

<h4>🔹🧠 BraTS 2021 – Run 2</h4>
<p align="center">
  <img src="BraTS_Run_2_metrics-01.jpg" width="70%">
</p>

<h4>🔹🧠 BraTS 2021 – Run 3</h4>
<p align="center">
  <img src="BraTS_Run_3_metrics-01.jpg" width="70%">
</p>

<hr>

<h3>❤️ ACDC – Independent Runs</h3>

<h4>🔹❤️ ACDC – Run 1</h4>
<p align="center">
  <img src="ACDC_Run_1_metrics.jpg" width="70%">
</p>

<h4>🔹❤️ ACDC – Run 2</h4>
<p align="center">
  <img src="ACDC_Run_2_metrics.jpg" width="70%">
</p>

<h4>🔹❤️ ACDC – Run 3</h4>
<p align="center">
  <img src="ACDC_Run_3_metrics.jpg" width="70%">
</p>

<hr>

<h3>📌 Key Observations</h3>

<ul>
  <li>Consistent convergence across all runs</li>
  <li>Minimal variation between training and validation curves</li>
  <li>Stable performance in DSC and IoU</li>
  <li>Gradual reduction in HD95, indicating improved boundary precision</li>
</ul>
