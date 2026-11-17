# Results Summary

## Robustness

|                 |   clean |   gauss_0.05 |   gauss_0.10 |   gauss_0.20 |   fgsm |    pgd |
|:----------------|--------:|-------------:|-------------:|-------------:|-------:|-------:|
| Baseline        |  0.6812 |       0.5906 |       0.4708 |       0.1979 | 0.0229 | 0      |
| Noise-Augmented |  0.65   |       0.6427 |       0.6542 |       0.524  | 0.025  | 0      |
| Adversarial     |  0.601  |       0.6021 |       0.5927 |       0.5208 | 0.1906 | 0.1302 |

## Denoising

|                 |    pgd |   pgd+denoise |   recovered |
|:----------------|-------:|--------------:|------------:|
| Baseline        | 0      |        0.2323 |      0.2323 |
| Noise-Augmented | 0      |        0.2375 |      0.2375 |
| Adversarial     | 0.1302 |        0.1854 |      0.0552 |

## Explanation metrics

|           |   avg_drop_% |   increase_in_confidence_% |   stability_corr |
|:----------|-------------:|---------------------------:|-----------------:|
| Grad-CAM  |       29.065 |                     25     |            0.826 |
| Score-CAM |        8.984 |                     33.333 |            0.979 |
