import secrets
import numpy as np

# =============================================================================
# Step 1: S-Box Construction (Möbius Transformation over GF(2^8))
# Paper Section 3.1.1, Table 2
# f(x) = (45x + 25) / (8x + 4) over GF(2^8), P(n) = x^8+x^6+x^5+x^4+1
# The exact 256-entry lookup table is taken directly from Table 2 of the paper.
# Note: The paper's symbolic formula produces a collision at output value 183
# (missing 147), so the authoritative source is the published table itself.
# =============================================================================
def build_sbox():
    """Return the S-box defined in Table 2 of the SuPOR paper (Möbius / GF(2^8))."""
    sbox_table = [
        #  0    1    2    3    4    5    6    7    8    9   10   11   12   13   14   15
        90, 212, 174, 70,  15,  39, 209,  87,   0, 127,  66, 173,  46, 231, 255, 189,
        # 16   17   18   19   20   21   22   23   24   25   26   27   28   29   30   31
       137, 115, 124, 183, 236, 248, 191, 198, 175, 226,  24, 155,  86, 223, 150, 220,
        # 32   33   34   35   36   37   38   39   40   41   42   43   44   45   46   47
       121, 216,  77, 144,  47, 208, 120, 172,  93, 135, 188,  35,  68,  82, 103, 234,
        # 48   49   50   51   52   53   54   55   56   57   58   59   60   61   62   63
       163,  74,  28,  49, 143, 184, 119,  25, 160, 247,  17,  88, 204, 105, 101, 186,
        # 64   65   66   67   68   69   70   71   72   73   74   75   76   77   78   79
        91, 116,  22, 193,  80, 166, 197, 246, 187, 130,  96,  97,   5, 149,  21, 237,
        # 80   81   82   83   84   85   86   87   88   89   90   91   92   93   94   95
       113, 131,  11,  13,  34, 219,  69,  57,  58, 239, 123, 207,  29,  72, 138, 106,
        # 96   97   98   99  100  101  102  103  104  105  106  107  108  109  110  111
       200,  26, 170, 118, 146, 228,   9, 217, 134,  33, 148, 202, 240,  40, 125,   7,
        #112  113  114  115  116  117  118  119  120  121  122  123  124  125  126  127
       107,  43,  84,  16, 179, 222,   3, 182, 177,   6, 159, 111,  44,  55, 249,  89,
        #128  129  130  131  132  133  134  135  136  137  138  139  140  141  142  143
       213,  73, 136, 181,  51,  41,  32,  12,  27, 141, 224,  19, 199, 110, 142, 151,
        #144  145  146  147  148  149  150  151  152  153  154  155  156  157  158  159
        99,  20, 251,  30,  61, 133,  36,   1,  95, 158,  83, 227,  56,  92, 168, 129,
        #160  161  162  163  164  165  166  167  168  169  170  171  172  173  174  175
       242, 117,  59, 169,  31, 165, 162, 132, 122,  98, 180, 229,   2,  67, 190, 140,
        #176  177  178  179  180  181  182  183  184  185  186  187  188  189  190  191
        48, 221, 192, 201,  81, 178, 250, 241, 147,  79, 253,   8, 164,  53, 102,  65,
        #192  193  194  195  196  197  198  199  200  201  202  203  204  205  206  207
       195,   4, 206, 238, 114, 232, 100,  63, 176,  50, 254, 161,  38, 210, 128,  78,
        #208  209  210  211  212  213  214  215  216  217  218  219  220  221  222  223
       185, 157,  85,  62,  37, 225, 145, 214, 215, 139, 156,  71,  18, 108, 211, 194,
        #224  225  226  227  228  229  230  231  232  233  234  235  236  237  238  239
       196,  64, 152,  75, 112, 233, 218,  23, 235, 244, 104, 153, 167,  60, 109, 203,
        #240  241  242  243  244  245  246  247  248  249  250  251  252  253  254  255
       126, 205,  76,  10,  54,  94, 154,  52, 245, 230,  45,  14, 243, 252, 171,  42,
    ]
    return np.array(sbox_table, dtype=np.uint8)

def build_inverse_sbox(sbox):
    """Derive the inverse S-box for decryption (reverse lookup)."""
    inv = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        inv[sbox[i]] = i
    return inv

# Pre-compute at module load
SBOX     = build_sbox()
INV_SBOX = build_inverse_sbox(SBOX)

def substitute_pixels(frame, sbox):
    """
    Step 1 – Pixel Substitution.
    Replaces every pixel value using the S-Box lookup table.
    Paper Pseudocode (Alg. 1): frame_pixels[i] ← pixel[frame_pixels[i]]
    Works on any shaped uint8 numpy array.
    """
    return sbox[frame]


# =============================================================================
# Step 2: Pixel Permutation
# Paper Section 3.2, Eq. (10)
# "The order and pattern of the substituted pixels were scattered without
#  repetition." — P_r(n, q) with n=q=len(pixels), i.e. a full permutation.
# Pseudocode: ind ← permutation(len(pix)); shuffle ← pix[ind].astype(uint8)
# The key is used as the PRNG seed so the same permutation is reproducible
# during decryption (simplest interpretation consistent with the one-time-key
# design of the paper).
# =============================================================================
def permute_pixels(flat_pixels, key):
    """
    Step 2 – Pixel Permutation.
    Scatter all pixels without repetition using a full P_r(n,n) permutation
    seeded by the encryption key.
    """
    rng = np.random.default_rng(key % (2**32))   # SeedSequence expects ≤ 128-bit
    ind = rng.permutation(len(flat_pixels))
    return flat_pixels[ind].astype(np.uint8)

def inverse_permute_pixels(shuffled, key):
    """
    Step 2 – Inverse Permutation for decryption.
    Reconstructs the original order by un-applying the same index permutation.
    """
    rng = np.random.default_rng(key % (2**32))
    ind = rng.permutation(len(shuffled))
    unshuffled = np.zeros_like(shuffled)
    # Place values back at original positions
    unshuffled[ind] = shuffled
    return unshuffled


# =============================================================================
# Step 3: Key Generation & Pixel XOR
# Paper Section 3.3, Eq. (11), (13)
# Key: k1 ⊕ k2 where k1, k2 are independent 64-bit CSPRNG values (secrets module).
# XOR: each pixel value p is XORed with key → pXOR = f_per ⊕ key
# Pseudocode: shuffle[index] ← values ⊕ key  (for each pixel vs. same key)
# The key is a 64-bit integer; pixel values are 8-bit. The paper applies the
# same 64-bit key integer to each pixel, which reduces to (key & 0xFF) per
# pixel because XOR is bit-level. Equivalently, cycling the 8 key-bytes across
# the array is identical. We use cyclic byte expansion (matches byte-array intent).
# =============================================================================
def generate_key():
    """
    Step 3 – CSPRNG Key Generation.
    Eq. (11): key = secrets.randbits(64) ⊕ secrets.randbits(64)
    Returns a single 64-bit integer.
    """
    k1 = secrets.randbits(64)
    k2 = secrets.randbits(64)
    return k1 ^ k2

def xor_pixels(flat_pixels, key):
    """
    Step 3 – Pixel XOR.
    Eq. (13): pXOR = f_per ⊕ key; result converted to byte array.
    The 64-bit key is split into 8 bytes and repeated cyclically to cover
    the full array. This is the simplest interpretation for visual data where
    each uint8 pixel is XORed with the corresponding key byte.
    XOR is its own inverse: calling this again with the same key undoes it.
    """
    key_bytes      = key.to_bytes(8, byteorder='big')
    key_tile       = np.resize(np.frombuffer(key_bytes, dtype=np.uint8), len(flat_pixels))
    return np.bitwise_xor(flat_pixels, key_tile)


# =============================================================================
# Step 4: Pixel Right Circular-Shift
# Paper Section 3.4, Eq. (14)
# Formula: f_cir → (n >> d) | (n << (bl - d))
# where n = byte value, d = 9, bl = byte-length = 8 bits.
# This is a BITWISE rotation of each individual byte value by d mod 8 = 1 bit.
# (Shifting 9 bits within an 8-bit byte wraps: effective shift = 9 mod 8 = 1.)
# The paper's bl=255 (length of byte array) appears as a typo/formatting issue;
# the formula structure (n>>d)|(n<<(...-d)) is the standard bitwise rotation.
# =============================================================================
def _ror8(value, d):
    """Right-rotate an 8-bit integer value by d bits."""
    d = d % 8
    return ((value >> d) | (value << (8 - d))) & 0xFF

def _rol8(value, d):
    """Left-rotate an 8-bit integer value by d bits (inverse of ror8)."""
    d = d % 8
    return ((value << d) | (value >> (8 - d))) & 0xFF

# Vectorised lookup tables for speed
_ROR_LUT = np.array([_ror8(v, 9) for v in range(256)], dtype=np.uint8)
_ROL_LUT = np.array([_rol8(v, 9) for v in range(256)], dtype=np.uint8)

def circular_shift(byte_array, d=9):
    """
    Step 4 – Right Circular-Shift.
    Eq. (14): Each byte value is right-rotated by d=9 bits (effective: 1 bit).
    Uses a pre-computed lookup table for efficiency.
    """
    return _ROR_LUT[byte_array]

def inverse_circular_shift(byte_array, d=9):
    """
    Step 4 – Inverse: Left-rotate each byte value to reverse Step 4.
    """
    return _ROL_LUT[byte_array]


# =============================================================================
# Step 5: Pixel Swap
# Paper Section 3.5, Eq. (15)
# f_swa → A, B = B, A
# Pseudocode: frame_swap[i], frame_swap[i+1] ← circular[i+1], circular[i]
# Swaps adjacent pairs (0,1), (2,3), (4,5), … throughout the array.
# Swap is its own inverse: applying it twice restores the original order.
# =============================================================================
def swap_pixels(byte_array):
    """
    Step 5 – Pixel Swap.
    Eq. (15): Swap every adjacent pair (i, i+1) for i = 0, 2, 4, …
    Works in-place on a copy using numpy reshaping for efficiency.
    If array length is odd the final element is left unchanged (no pair).
    """
    swapped = byte_array.copy()
    n = len(swapped)
    # Vectorised swap of all even/odd index pairs
    even = n // 2 * 2          # largest even ≤ n
    swapped[:even:2], swapped[1:even:2] = swapped[1:even:2].copy(), swapped[:even:2].copy()
    return swapped


# =============================================================================
# Full Encryption
# Paper Fig. 1, Algorithm 1, Section 3
# Order: Substitution → Permutation → XOR → Right-Circular-Shift → Swap
# =============================================================================
def supor_encrypt(frame, key):
    """
    Encrypt a frame with the SuPOR stream cipher.

    Args:
        frame : numpy.ndarray  uint8, any shape (H×W or H×W×C)
        key   : int            64-bit one-time key from generate_key()

    Returns:
        Encrypted frame of the same shape and dtype as input.
    """
    orig_shape = frame.shape

    # Step 1 – Substitution
    substituted = substitute_pixels(frame, SBOX)

    # Step 2 – Permutation (operate on flat array)
    flat = substituted.flatten()
    permuted = permute_pixels(flat, key)

    # Step 3 – XOR + convert to byte array
    xored = xor_pixels(permuted, key)

    # Step 4 – Right Circular-Shift of each byte value
    shifted = circular_shift(xored, d=9)

    # Step 5 – Swap adjacent bytes
    swapped = swap_pixels(shifted)

    return swapped.reshape(orig_shape)


# =============================================================================
# Full Decryption
# Paper Section 3.6: "the reverse of its encryption process"
# Reverse order: Swap → Left-Circular-Shift → XOR → Inverse-Permutation → Inverse-Substitution
# =============================================================================
def supor_decrypt(encrypted_frame, key):
    """
    Decrypt a SuPOR-encrypted frame.

    Args:
        encrypted_frame : numpy.ndarray  uint8, same shape as the original
        key             : int            the same 64-bit key used for encryption

    Returns:
        Decrypted frame (should be byte-identical to the original plaintext).
    """
    orig_shape = encrypted_frame.shape
    flat = encrypted_frame.flatten()

    # Reverse Step 5 – Swap (self-inverse)
    unswapped = swap_pixels(flat)

    # Reverse Step 4 – Left Circular-Shift
    unshifted = inverse_circular_shift(unswapped, d=9)

    # Reverse Step 3 – XOR (self-inverse)
    unxored = xor_pixels(unshifted, key)

    # Reverse Step 2 – Inverse Permutation
    unpermuted = inverse_permute_pixels(unxored, key)

    # Reverse Step 1 – Inverse Substitution
    decrypted = substitute_pixels(unpermuted, INV_SBOX)

    return decrypted.reshape(orig_shape)
