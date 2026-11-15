# Brag Plan: Adversarial Robustness of CNNs (long cut)

## Run note
The user asked for **at least 1 minute**, which overrides the default 15–25s brag length. Target: **72.5s**. The short cut is still in `brag-output/`.

## What is this app?
A CSI 5170 research project. It measures how a transfer-learned ResNet-50 on Tiny ImageNet breaks under Gaussian noise, FGSM and PGD, which defenses help, and whether Grad-CAM and Score-CAM stay trustworthy under attack.

## The angle
The model is wrong with total confidence: a goldfish becomes a **cougar (100%)**. The long cut keeps that hook, then follows the research story: how small the attack is, how random and aimed noise compare, three defenses (including the adversarial-training runs that failed before one worked), recovery, and which explanation method to trust. Every number comes from README.md / RESULTS_SUMMARY.md, and every image from `results/`.

## Hook
"This is a goldfish." over the real green-framed tile from `pgd_recovery.png`. At 3.70s (a strong beat) it flips red: `✗ cougar (100%)`.

## User flow worth showing
Clean image → attack → model fooled → defense → recovered with correct attention. This is the PGD recovery figure, rebuilt with its real tiles.

## Tone
- Preset: default
- Creative direction: dry lab-notebook comedy that becomes a clear research story
- Interpretation: clean reveals with long enough holds to read numbers; humor only from the real labels and the failed runs

## Format: landscape — 1920x1080
## Duration: 72.5s

## Visual identity
- Background #0f1b2d · text #f2f4f7 · report accent #1f3a5f · correct #1a9641 (text #5bd68a) · wrong #d7191c (text #ff7a7a)
- Display: Liberation Serif; labels: DejaVu Sans; numbers: Noto Sans Mono

## Audio direction
- Music: happy-beats-business-moves-vol-9 (114.8 BPM), bed 0.30, fade over the last 2s
- Cue guidance: preset JSON. Beat locks at 3.70 (flip), 30.01 (PGD bar), 43.70 (denoise note); reveals snap to the beat grid at ≥1 beat apart, and the full set holds afterward
- Audio-reactive: subtle RMS glow behind the content
- SFX: sparse and motion-matched: impact plus glitch on the flip, card slides for the wrong verdicts, soft drops for reveals, a quiet error buzz for the failed runs, a bell for the working run, and a bong on the title

## Storyboard (global times)
1. **Hook + attack — 0–12.6s.** "This is a goldfish." Flips to cougar at 3.70, then joins a row with syringe (6.86) and umbrella (7.92). "All three are goldfish." holds to 12.4.
2. **The attack, amplified — 12.6–19.5s.** Clean tile + amplified difference (×10) = attacked tile (`✗ cougar`). Line: "At most 8/255 per pixel. Invisible to you. Decisive to ResNet-50."
3. **The project — 19.5–26.0s.** "How fragile is a ResNet-50?" Five pipeline chips on alternate beats: Tiny ImageNet → ResNet-50 → Attacks → Defenses → Explain.
4. **RQ1 — 26.0–35.0s.** Six baseline accuracy bars: Clean 68.1 · σ=0.05 59.1 · σ=0.10 47.1 · σ=0.20 19.8 · FGSM 2.3 · PGD 0.0 (beat-locked 30.01). Then: "Random noise has to be loud. Aimed noise only has to be right."
5. **Defense 1, noise augmentation — 35.0–41.0s.** σ=0.20: 19.8% → 52.4%. PGD: 0.0% → 0.0%. "No adversarial robustness."
6. **Defense 2, test-time denoising — 41.0–46.5s.** PGD + Gaussian filter: 0.0% → 22.8%. Note (43.70): "Upper bound: the attack doesn't know about the filter."
7. **Defense 3, adversarial training, take three — 46.5–55.0s.** Run 1 ✗ (BN trainable: clean 39% → 4%), Run 2 ✗ (top block only: brittle, σ=0.05 → 2.7%), Run 3 ✓ (full backbone, ε-curriculum: clean ≈62%, PGD ≈13%).
8. **Recovery — 55.0–61.0s.** Attacked goldfish, defended model: `✓ goldfish (95%)` + Grad-CAM + 27/28 recovered (labelled PGD transfer attacks).
9. **RQ3, explanations — 61.0–68.5s.** Grad-CAM vs Score-CAM tiles; Avg. Drop 29.0% vs 9.0%, Stability 0.826 vs 0.979. "Score-CAM is more faithful and more stable, at ~256 forward passes per image."
10. **Outro — 68.5–72.5s.** Title card.

## Share copy (draft)
A ResNet-50 was 100% sure this goldfish was a cougar. One minute on how fragile CNNs are, which defenses actually help, and why Score-CAM beats Grad-CAM.
