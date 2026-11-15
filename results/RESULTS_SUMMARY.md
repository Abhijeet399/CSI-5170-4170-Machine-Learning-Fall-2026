# Results Summary

## Robustness

|                 |   clean |   gauss_0.05 |   gauss_0.10 |   gauss_0.20 |   fgsm |   pgd |
|:----------------|--------:|-------------:|-------------:|-------------:|-------:|------:|
| Baseline        |  0.6812 |       0.5906 |       0.4708 |       0.1979 | 0.0229 |     0 |
| Noise-Augmented |  0.65   |       0.6427 |       0.6542 |       0.524  | 0.025  |     0 |
| Adversarial     |  0.7083 |       0.0271 |       0.0031 |       0.001  | 0.0073 |     0 |

## Denoising

|                 |   pgd |   pgd+denoise |   recovered |
|:----------------|------:|--------------:|------------:|
| Baseline        |     0 |        0.2281 |      0.2281 |
| Noise-Augmented |     0 |        0.2323 |      0.2323 |
| Adversarial     |     0 |        0.0427 |      0.0427 |

## Explanation metrics

|           |   avg_drop_% |   increase_in_confidence_% |   stability_corr |
|:----------|-------------:|---------------------------:|-----------------:|
| Grad-CAM  |       29.039 |                     25     |            0.826 |
| Score-CAM |        9.042 |                     33.333 |            0.979 |
