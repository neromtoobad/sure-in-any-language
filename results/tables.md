## Table 1. Accuracy, confidence and the gap between them

| Model | Lang | Accuracy | Confidence | Gap (pp) [95% CI] | Gap minus English gap [95% CI] | ECE | AUROC | Answers at ≥90% | Wrong among those |
|---|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global 3.35B (n=300) | English | 82% | 88% | +6 [+2, +10] | – | 0.07 | 0.83 | 67% | 6% (n=202) |
| Tiny Aya Global 3.35B (n=300) | Swahili | 75% | 82% | +8 [+3, +12] | +2 [-3, +7] | 0.09 | 0.74 | 51% | 12% (n=152) |
| Tiny Aya Global 3.35B (n=300) | Hausa | 53% | 65% | +13 [+8, +18] | +7 [+1, +12] | 0.13 | 0.73 | 12% | 11% (n=35) |
| Tiny Aya Global 3.35B (n=300) | Igbo | 42% | 62% | +20 [+14, +26] | +14 [+8, +20] | 0.20 | 0.65 | 10% | 32% (n=31) |
| Tiny Aya Global 3.35B (n=300) | Yoruba | 43% | 64% | +20 [+15, +26] | +14 [+8, +20] | 0.21 | 0.63 | 9% | 22% (n=27) |
| Gemma 3 4B (n=300) | English | 84% | 99% | +15 [+11, +19] | – | 0.17 | 0.79 | 97% | 16% (n=290) |
| Gemma 3 4B (n=300) | Swahili | 62% | 99% | +37 [+32, +43] | +22 [+16, +28] | 0.38 | 0.69 | 96% | 38% (n=289) |
| Gemma 3 4B (n=300) | Hausa | 44% | 98% | +54 [+48, +59] | +39 [+32, +45] | 0.54 | 0.63 | 95% | 55% (n=284) |
| Gemma 3 4B (n=300) | Igbo | 35% | 98% | +63 [+57, +68] | +48 [+41, +54] | 0.63 | 0.50 | 94% | 65% (n=281) |
| Gemma 3 4B (n=300) | Yoruba | 35% | 98% | +63 [+57, +68] | +48 [+41, +54] | 0.63 | 0.57 | 95% | 64% (n=284) |
| Llama 3.2 3B (n=300) | English | 82% | 81% | -1 [-5, +3] | – | 0.04 | 0.84 | 53% | 4% (n=160) |
| Llama 3.2 3B (n=300) | Swahili | 53% | 55% | +2 [-3, +7] | +3 [-3, +9] | 0.03 | 0.73 | 13% | 8% (n=39) |
| Llama 3.2 3B (n=300) | Hausa | 35% | 43% | +8 [+3, +13] | +9 [+2, +15] | 0.08 | 0.58 | 1% | 0% (n=2) |
| Llama 3.2 3B (n=300) | Igbo | 32% | 40% | +8 [+3, +13] | +9 [+2, +15] | 0.08 | 0.57 | 0% | 0% (n=1) |
| Llama 3.2 3B (n=300) | Yoruba | 32% | 41% | +9 [+3, +14] | +10 [+3, +16] | 0.09 | 0.53 | 1% | 50% (n=2) |
| Phi-4-mini 3.8B (n=300) | English | 89% | 94% | +5 [+2, +8] | – | 0.06 | 0.87 | 82% | 4% (n=245) |
| Phi-4-mini 3.8B (n=300) | Swahili | 62% | 76% | +14 [+9, +19] | +9 [+3, +15] | 0.14 | 0.77 | 36% | 16% (n=109) |
| Phi-4-mini 3.8B (n=300) | Hausa | 40% | 55% | +15 [+10, +21] | +11 [+4, +17] | 0.16 | 0.58 | 6% | 26% (n=19) |
| Phi-4-mini 3.8B (n=300) | Igbo | 39% | 53% | +14 [+9, +20] | +9 [+3, +16] | 0.15 | 0.58 | 3% | 22% (n=9) |
| Phi-4-mini 3.8B (n=300) | Yoruba | 35% | 51% | +17 [+11, +22] | +12 [+6, +18] | 0.17 | 0.59 | 2% | 29% (n=7) |
| Tiny Aya Earth 3.35B (4-bit) (n=300) | English | 85% | 90% | +5 [+2, +9] | – | 0.07 | 0.83 | 73% | 7% (n=219) |
| Tiny Aya Earth 3.35B (4-bit) (n=300) | Swahili | 73% | 82% | +9 [+5, +14] | +4 [-1, +9] | 0.09 | 0.79 | 50% | 11% (n=151) |
| Tiny Aya Earth 3.35B (4-bit) (n=300) | Hausa | 56% | 69% | +12 [+7, +17] | +7 [+2, +12] | 0.12 | 0.74 | 21% | 12% (n=64) |
| Tiny Aya Earth 3.35B (4-bit) (n=300) | Igbo | 46% | 64% | +19 [+13, +24] | +14 [+7, +19] | 0.19 | 0.64 | 16% | 34% (n=47) |
| Tiny Aya Earth 3.35B (4-bit) (n=300) | Yoruba | 44% | 64% | +19 [+14, +25] | +14 [+8, +20] | 0.20 | 0.65 | 17% | 29% (n=52) |

## Table 2. One confidence cut-off, five languages (target: at most 10% wrong among answered)

| Model | Lang | English cut-off: answered | English cut-off: wrong | Own cut-off: answered | Own cut-off: wrong | ECE raw | ECE English temperature | ECE own temperature |
|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global 3.35B | English | 78% | 9% | 78% | 9% | 0.07 | 0.03 | 0.03 |
| Tiny Aya Global 3.35B | Swahili | 65% | 16% | 42% | 11% | 0.09 | 0.04 | 0.05 |
| Tiny Aya Global 3.35B | Hausa | 29% | 25% | 9% | 21% | 0.13 | 0.05 | 0.06 |
| Tiny Aya Global 3.35B | Igbo | 22% | 42% | 2% | 40% | 0.20 | 0.13 | 0.05 |
| Tiny Aya Global 3.35B | Yoruba | 25% | 43% | 3% | 44% | 0.21 | 0.12 | 0.04 |
| Gemma 3 4B | English | 85% | 12% | 85% | 12% | 0.17 | 0.02 | 0.02 |
| Gemma 3 4B | Swahili | 79% | 32% | 0% | – | 0.38 | 0.22 | 0.08 |
| Gemma 3 4B | Hausa | 68% | 50% | 0% | – | 0.54 | 0.37 | 0.03 |
| Gemma 3 4B | Igbo | 63% | 66% | 0% | – | 0.63 | 0.46 | 0.00 |
| Gemma 3 4B | Yoruba | 59% | 60% | 0% | – | 0.63 | 0.44 | 0.03 |
| Llama 3.2 3B | English | 79% | 11% | 79% | 11% | 0.04 | 0.05 | 0.05 |
| Llama 3.2 3B | Swahili | 36% | 27% | 16% | 15% | 0.03 | 0.03 | 0.04 |
| Llama 3.2 3B | Hausa | 12% | 42% | 1% | 0% | 0.08 | 0.08 | 0.03 |
| Llama 3.2 3B | Igbo | 8% | 48% | 1% | 33% | 0.08 | 0.07 | 0.04 |
| Llama 3.2 3B | Yoruba | 10% | 55% | 1% | 50% | 0.09 | 0.09 | 0.04 |
| Phi-4-mini 3.8B | English | 97% | 10% | 97% | 10% | 0.06 | 0.02 | 0.02 |
| Phi-4-mini 3.8B | Swahili | 80% | 30% | 33% | 16% | 0.14 | 0.05 | 0.04 |
| Phi-4-mini 3.8B | Hausa | 43% | 56% | 2% | 17% | 0.16 | 0.07 | 0.05 |
| Phi-4-mini 3.8B | Igbo | 42% | 54% | 3% | 25% | 0.15 | 0.06 | 0.05 |
| Phi-4-mini 3.8B | Yoruba | 36% | 55% | 2% | 57% | 0.17 | 0.08 | 0.02 |

## Table 3. Signals that need no labels in the target language

| Model | Lang | Flips when options are reversed | Accuracy stable / flipped | Same answer as in English | Accuracy agree / disagree | Stable and agrees: answered | wrong | AUROC confidence | AUROC yes/no self-check | AUROC stated number | AUROC combined |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tiny Aya Global 3.35B | English | 19% | 93% / 37% | – | – / – | – | – | 0.83 | 0.46 | 0.55 | – |
| Tiny Aya Global 3.35B | Swahili | 23% | 85% / 41% | 77% | 89% / 28% | 65% | 7% | 0.74 | 0.51 | 0.51 | 0.88 |
| Tiny Aya Global 3.35B | Hausa | 39% | 70% / 25% | 61% | 84% / 5% | 46% | 9% | 0.73 | 0.55 | 0.52 | 0.90 |
| Tiny Aya Global 3.35B | Igbo | 41% | 53% / 26% | 48% | 85% / 3% | 35% | 12% | 0.65 | 0.48 | 0.50 | 0.88 |
| Tiny Aya Global 3.35B | Yoruba | 49% | 57% / 29% | 45% | 88% / 7% | 29% | 8% | 0.63 | 0.48 | 0.48 | 0.86 |
| Gemma 3 4B | English | 17% | 91% / 47% | – | – / – | – | – | 0.79 | 0.62 | 0.50 | – |
| Gemma 3 4B | Swahili | 29% | 73% / 33% | 66% | 87% / 13% | 54% | 9% | 0.69 | 0.51 | 0.48 | 0.88 |
| Gemma 3 4B | Hausa | 39% | 54% / 30% | 44% | 89% / 8% | 33% | 9% | 0.63 | 0.54 | 0.56 | 0.86 |
| Gemma 3 4B | Igbo | 51% | 39% / 31% | 36% | 86% / 7% | 20% | 12% | 0.50 | 0.50 | 0.52 | 0.78 |
| Gemma 3 4B | Yoruba | 58% | 40% / 32% | 38% | 84% / 5% | 19% | 16% | 0.57 | 0.52 | 0.51 | 0.83 |
| Llama 3.2 3B | English | 16% | 89% / 48% | – | – / – | – | – | 0.84 | 0.74 | 0.67 | – |
| Llama 3.2 3B | Swahili | 50% | 69% / 37% | 57% | 86% / 9% | 36% | 9% | 0.73 | 0.64 | 0.58 | 0.86 |
| Llama 3.2 3B | Hausa | 66% | 44% / 30% | 36% | 83% / 8% | 16% | 16% | 0.58 | 0.54 | 0.52 | 0.80 |
| Llama 3.2 3B | Igbo | 63% | 31% / 33% | 32% | 91% / 5% | 10% | 3% | 0.57 | 0.53 | 0.51 | 0.80 |
| Llama 3.2 3B | Yoruba | 72% | 36% / 31% | 31% | 90% / 6% | 10% | 7% | 0.53 | 0.51 | 0.51 | 0.81 |
| Phi-4-mini 3.8B | English | 6% | 93% / 29% | – | – / – | – | – | 0.87 | 0.72 | 0.63 | – |
| Phi-4-mini 3.8B | Swahili | 23% | 70% / 34% | 59% | 97% / 11% | 52% | 1% | 0.77 | 0.57 | 0.52 | 0.92 |
| Phi-4-mini 3.8B | Hausa | 41% | 40% / 41% | 38% | 94% / 7% | 22% | 5% | 0.58 | 0.56 | 0.51 | 0.78 |
| Phi-4-mini 3.8B | Igbo | 43% | 40% / 38% | 36% | 95% / 7% | 21% | 2% | 0.58 | 0.50 | 0.50 | 0.79 |
| Phi-4-mini 3.8B | Yoruba | 40% | 36% / 32% | 34% | 90% / 6% | 22% | 12% | 0.59 | 0.57 | 0.49 | 0.81 |

## Table 4. What the model says about itself

| Model | Lang | Accuracy | Stated confidence (mean) | Most common stated number (share) | Says "Yes, correct" | Says Yes when wrong |
|---|---|---|---|---|---|---|
| Tiny Aya Global 3.35B | English | 82% | 68% | 65 (80%) | 55% | 59% |
| Tiny Aya Global 3.35B | Swahili | 75% | 67% | 65 (86%) | 53% | 53% |
| Tiny Aya Global 3.35B | Hausa | 53% | 67% | 65 (89%) | 34% | 30% |
| Tiny Aya Global 3.35B | Igbo | 42% | 67% | 65 (88%) | 5% | 3% |
| Tiny Aya Global 3.35B | Yoruba | 43% | 66% | 65 (93%) | 52% | 52% |
| Gemma 3 4B | English | 84% | 95% | 95 (100%) | 99% | 98% |
| Gemma 3 4B | Swahili | 62% | 94% | 95 (94%) | 100% | 100% |
| Gemma 3 4B | Hausa | 44% | 94% | 95 (84%) | 99% | 98% |
| Gemma 3 4B | Igbo | 35% | 94% | 95 (87%) | 97% | 97% |
| Gemma 3 4B | Yoruba | 35% | 93% | 95 (78%) | 97% | 95% |
| Llama 3.2 3B | English | 82% | 51% | 100 (38%) | 28% | 8% |
| Llama 3.2 3B | Swahili | 53% | 19% | 0 (54%) | 3% | 1% |
| Llama 3.2 3B | Hausa | 35% | 4% | 0 (90%) | 0% | 0% |
| Llama 3.2 3B | Igbo | 32% | 2% | 0 (95%) | 0% | 0% |
| Llama 3.2 3B | Yoruba | 32% | 1% | 0 (99%) | 0% | 0% |
| Phi-4-mini 3.8B | English | 89% | 98% | 100 (54%) | 97% | 97% |
| Phi-4-mini 3.8B | Swahili | 62% | 95% | 95 (94%) | 98% | 97% |
| Phi-4-mini 3.8B | Hausa | 40% | 95% | 95 (98%) | 99% | 98% |
| Phi-4-mini 3.8B | Igbo | 39% | 95% | 95 (98%) | 99% | 99% |
| Phi-4-mini 3.8B | Yoruba | 35% | 95% | 95 (99%) | 99% | 99% |

## Table 5. Tiny Aya Global at different precisions

| Precision | Lang | Accuracy | Confidence | Gap (pp) [95% CI] | ECE | AUROC | Flips when reversed |
|---|---|---|---|---|---|---|---|
| 8-bit (n=300) | English | 83% | 91% | +8 [+4, +11] | 0.08 | 0.83 | 15% |
| 8-bit (n=300) | Swahili | 75% | 83% | +8 [+3, +12] | 0.10 | 0.75 | 20% |
| 8-bit (n=300) | Hausa | 56% | 66% | +10 [+5, +15] | 0.11 | 0.72 | 37% |
| 8-bit (n=300) | Igbo | 44% | 64% | +20 [+14, +25] | 0.20 | 0.64 | 39% |
| 8-bit (n=300) | Yoruba | 43% | 64% | +22 [+16, +27] | 0.22 | 0.67 | 44% |
| 4-bit (n=300) | English | 82% | 88% | +6 [+2, +10] | 0.07 | 0.83 | 19% |
| 4-bit (n=300) | Swahili | 75% | 82% | +8 [+3, +12] | 0.09 | 0.74 | 23% |
| 4-bit (n=300) | Hausa | 53% | 65% | +13 [+8, +18] | 0.13 | 0.73 | 39% |
| 4-bit (n=300) | Igbo | 42% | 62% | +20 [+14, +26] | 0.20 | 0.65 | 41% |
| 4-bit (n=300) | Yoruba | 43% | 64% | +20 [+15, +26] | 0.21 | 0.63 | 49% |
| 3-bit (n=300) | English | 81% | 88% | +8 [+4, +12] | 0.09 | 0.83 | 21% |
| 3-bit (n=300) | Swahili | 70% | 83% | +13 [+9, +18] | 0.13 | 0.79 | 26% |
| 3-bit (n=300) | Hausa | 57% | 67% | +10 [+5, +16] | 0.11 | 0.68 | 38% |
| 3-bit (n=300) | Igbo | 44% | 64% | +20 [+15, +25] | 0.20 | 0.69 | 45% |
| 3-bit (n=300) | Yoruba | 43% | 65% | +22 [+17, +28] | 0.23 | 0.67 | 46% |
| 2-bit (n=300) | English | 64% | 76% | +11 [+6, +16] | 0.12 | 0.77 | 39% |
| 2-bit (n=300) | Swahili | 55% | 66% | +11 [+5, +16] | 0.11 | 0.67 | 54% |
| 2-bit (n=300) | Hausa | 41% | 56% | +15 [+9, +20] | 0.15 | 0.66 | 61% |
| 2-bit (n=300) | Igbo | 28% | 55% | +27 [+22, +32] | 0.27 | 0.58 | 72% |
| 2-bit (n=300) | Yoruba | 34% | 49% | +15 [+10, +21] | 0.15 | 0.59 | 67% |
