import numpy as np
import urllib.request
from PIL import Image
import os
import supor
import metrics


# =============================================================================
# Image Preparation
# =============================================================================
def load_image(path, mode="RGB"):
    """
    Load an image from disk and convert it to the desired mode.
    Args:
        path : str   path to the image file
        mode : str   'RGB' for colour testing, 'L' for greyscale
    Returns:
        numpy.ndarray of dtype uint8
    """
    img = Image.open(path).convert(mode)
    return np.array(img, dtype=np.uint8)

def download_sample_image(filename="sample.png"):
    """Download a sample PNG image if not already present locally."""
    if not os.path.exists(filename):
        # Publicly available Giza Pyramids image (Wikimedia Commons)
        url = (
            "https://upload.wikimedia.org/wikipedia/commons/c/c0/"
            "Giza_Pyramids_-%D8%A3%D9%87%D8%B1%D8%A7%D9%85%D8%A7%D8%AA"
            "_%D8%A7%D9%84%D8%AC%D9%8A%D8%B2%D8%A9.png"
        )
        print(f"Downloading sample image → {filename} …")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(filename, "wb") as f:
            f.write(resp.read())


# =============================================================================
# Main
# =============================================================================
def main():
    # ------------------------------------------------------------------
    # 1. Prepare frame
    # ------------------------------------------------------------------
    download_sample_image("sample.png")

    IMAGE_MODE = "RGB"          # Change to "L" for greyscale
    RESIZE     = (256, 256)     # Resize for consistent & fast testing

    frame = load_image("sample.png", mode=IMAGE_MODE)
    if RESIZE:
        img_pil = Image.fromarray(frame)
        img_pil = img_pil.resize(RESIZE)
        frame = np.array(img_pil, dtype=np.uint8)

    print(f"Original frame  shape : {frame.shape}  dtype : {frame.dtype}")

    # Save original for visual reference
    Image.fromarray(frame).save("original.png")

    # ------------------------------------------------------------------
    # 2. Key generation (Paper Sec. 3.3.1 / Eq. 11)
    # ------------------------------------------------------------------
    key = supor.generate_key()
    print(f"64-bit one-time key   : {key}")

    # ------------------------------------------------------------------
    # 3. Encryption (Paper Sec. 3, Algorithm 1)
    # ------------------------------------------------------------------
    print("Encrypting …")
    encrypted_frame = supor.supor_encrypt(frame, key)

    # Must save as lossless PNG — any lossy format (JPEG) would corrupt
    # the cipher-text and prevent correct decryption.
    Image.fromarray(encrypted_frame).save("encrypted.png")
    print("Saved encrypted image → encrypted.png")

    # ------------------------------------------------------------------
    # 4. Decryption (Paper Sec. 3.6 — reverse of encryption)
    # ------------------------------------------------------------------
    print("Decrypting …")
    decrypted_frame = supor.supor_decrypt(encrypted_frame, key)
    Image.fromarray(decrypted_frame).save("decrypted.png")
    print("Saved decrypted image → decrypted.png")

    # ------------------------------------------------------------------
    # 5. Correctness check
    # ------------------------------------------------------------------
    if np.array_equal(frame, decrypted_frame):
        print("\n✓  SUCCESS — Decrypted frame is a perfect bit-exact match of the original.")
    else:
        diff_count = np.sum(frame != decrypted_frame)
        print(f"\n✗  ERROR   — {diff_count} pixels differ between original and decrypted frame.")

    # ------------------------------------------------------------------
    # 6. Evaluation metrics (Paper Sec. 5.2.6, 5.3.2, 5.3.4)
    # ------------------------------------------------------------------
    print("\n--- Evaluation Metrics (Paper Definitions) ---")

    # Entropy (Sec 5.3.2, Eq. 20)
    entropy_enc = metrics.compute_entropy(encrypted_frame)
    entropy_dec = metrics.compute_entropy(frame)

    # NPCR & UACI (Sec 5.2.6, Eq. 18-19)
    # Flip exactly one LSB of one pixel, keep the same key
    diff_frame            = frame.copy()
    diff_frame[0, 0, 0]   = diff_frame[0, 0, 0] ^ 1   # flip 1 bit
    diff_encrypted         = supor.supor_encrypt(diff_frame, key)

    npcr = metrics.compute_npcr(encrypted_frame, diff_encrypted)
    uaci = metrics.compute_uaci(encrypted_frame, diff_encrypted)

    # Spatial adjacent-pixel correlation (Sec 5.3.4, Eq. 23)
    corr_orig = metrics.compute_adjacent_correlation(frame)
    corr_enc  = metrics.compute_adjacent_correlation(encrypted_frame)

    report_lines = [
        "SuPOR Cipher – Evaluation Report",
        "=" * 42,
        f"Frame shape           : {frame.shape}",
        f"Key (64-bit)          : {key}",
        "",
        "  [Entropy – Eq. 20]",
        f"  Original  : {entropy_dec:.4f} bits  (ideal for random ≈ 8.0)",
        f"  Encrypted : {entropy_enc:.4f} bits  (ideal ≥ 7.9)",
        "",
        "  [Differential Attack – Eq. 18 / 19]",
        f"  NPCR      : {npcr:.4f} %  (ideal ≥ 99%)",
        f"  UACI      : {uaci:.4f} %  (ideal ≥ 33%)",
        "",
        "  [Adjacent-Pixel Correlation – Eq. 23]",
        f"  Original  : {corr_orig:.4f}  (high correlation expected)",
        f"  Encrypted : {corr_enc:.4f}  (ideal ≈ 0)",
        "=" * 42,
    ]

    report = "\n".join(report_lines)
    print(report)

    with open("metrics.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print("\nSaved metrics report → metrics.txt")

    # Histogram (Sec 5.3.1)
    metrics.plot_histograms(frame, encrypted_frame, filename="histograms.png")


if __name__ == "__main__":
    main()
