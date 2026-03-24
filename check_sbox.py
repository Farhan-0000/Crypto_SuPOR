def gf_mult(a, b, poly=0x171):
    p = 0
    for _ in range(8):
        if b & 1: p ^= a
        hi = a & 0x80
        a <<= 1
        if hi: a ^= poly
        b >>= 1
    return p & 0xFF

def gf_inv(a, poly=0x171):
    if a == 0: return 0
    for i in range(1, 256):
        if gf_mult(a, i, poly) == 1:
            return i
    return 0

sbox = []
for x in range(256):
    num = (gf_mult(45, x) ^ 25)
    den = (gf_mult(8, x) ^ 4)
    if den == 0:
        val = num
    else:
        val = gf_mult(num, gf_inv(den))
    sbox.append(val)

print("Constructed S-Box Head:")
print(sbox[:16])

