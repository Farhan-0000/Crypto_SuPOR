import numpy as np
import math
import matplotlib.pyplot as plt


def compute_entropy(image):
    """
    Shannon entropy of an image frame.
    Paper Section 5.3.2, Eq. (20):
        H(Y) = -sum_{i=0}^{x-1} p(i) * log2(p(i))
    where x = number of grey levels (256 for 8-bit), p(i) = probability of
    grey level i.  Entropy is computed on the grayscale (flattened) histogram;
    the ideal maximum for a random 8-bit image is 8.0 bits.
    range=(0, 256) ensures all 256 values [0…255] are captured correctly.
    """
    histogram, _ = np.histogram(image.flatten(), bins=256, range=(0, 256))
    total = float(np.sum(histogram))
    probabilities = histogram / total
    entropy = -np.sum([p * math.log2(p) for p in probabilities if p > 0])
    return entropy


def compute_npcr(img1, img2):
    """
    Number of Pixel Changing Rate (NPCR).
    Paper Section 5.2.6, Eq. (18):
        NPCR(fe, fe*) = [sum P(i,j) / T] * 100
        P(i,j) = 0  if fe(i,j) == fe*(i,j),  else 1
    where T = total pixel count.
    Both images must have identical shape.  For colour images all channels
    are compared element-wise (same as the paper's per-pixel definition).
    Ideal: ≥ 99%.
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    if img1.shape != img2.shape:
        raise ValueError("NPCR: images must have the same shape.")
    diff = (img1 != img2).astype(np.float64)
    return np.mean(diff) * 100.0


def compute_uaci(img1, img2):
    """
    Unified Average Changing Intensity (UACI).
    Paper Section 5.2.6, Eq. (19):
        UACI(fe, fe*) = (1/T) * [sum |fe(i,j) - fe*(i,j)| / HPV] * 100
    where HPV = 255 (highest pixel value).
    Ideal: ≥ 33%.
    """
    img1 = np.asarray(img1, dtype=np.float64)
    img2 = np.asarray(img2, dtype=np.float64)
    if img1.shape != img2.shape:
        raise ValueError("UACI: images must have the same shape.")
    diff = np.abs(img1 - img2) / 255.0
    return np.mean(diff) * 100.0


def compute_adjacent_correlation(image, num_samples=5000):
    """
    Pearson correlation coefficient between horizontally adjacent pixel pairs.
    Paper Section 5.3.4, Eq. (23):
        r = sum((xi - x̄)(yi - ȳ)) / sqrt(sum(xi-x̄)^2 * sum(yi-ȳ)^2)
    For a colour image the first channel (R) is used, matching the paper's
    greyscale histogram approach for video frame analysis.
    Adjacent pixel pairs: (x, y) and (x+1, y) across rows (horizontal).
    Ideal encrypted value: r ≈ 0 (no correlation).
    """
    # Work on 2D (grayscale view): if colour, use luminance channel
    if image.ndim == 3:
        gray = image[:, :, 0]     # use R channel for consistency
    else:
        gray = image

    h, w = gray.shape
    rng = np.random.default_rng(42)   # fixed seed for reproducibility

    # Sample random horizontal pairs (x, y) vs (x+1, y)
    max_pairs = (h * (w - 1))
    n = min(num_samples, max_pairs)

    rows = rng.integers(0, h, size=n)
    cols = rng.integers(0, w - 1, size=n)

    x_vals = gray[rows, cols].astype(np.float64)
    y_vals = gray[rows, cols + 1].astype(np.float64)

    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
    return correlation


def plot_histograms(original, encrypted, filename="histograms.png"):
    """
    Histogram analysis as in Paper Section 5.3.1 / Table 13.
    Plots pixel intensity distributions for original and encrypted frames
    side by side.  A well-encrypted frame should have a flat (uniform)
    histogram, distinct from the non-uniform original.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].hist(original.flatten(), bins=256, range=(0, 256),
                 color='steelblue', alpha=0.85, edgecolor='none')
    axes[0].set_title('Original Frame – Histogram', fontsize=13)
    axes[0].set_xlabel('Pixel Intensity (0–255)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim([0, 255])

    axes[1].hist(encrypted.flatten(), bins=256, range=(0, 256),
                 color='firebrick', alpha=0.85, edgecolor='none')
    axes[1].set_title('Encrypted Frame – Histogram', fontsize=13)
    axes[1].set_xlabel('Pixel Intensity (0–255)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim([0, 255])

    plt.suptitle('SuPOR Histogram Analysis', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved histogram plot → {filename}")
