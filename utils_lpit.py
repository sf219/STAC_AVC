import numpy as np
import io
import math
from numba import njit


def unifgrid(N):
    H = N
    W = N
    d1 = np.ones(H * W - 1)
    d1[H - 1 :: H] = 0
    dN = np.ones(H * W - H)
    W = np.diag(d1, -1) + np.diag(d1, 1) + np.diag(dN, -H) + np.diag(dN, H)
    L = np.diag(np.sum(W, axis=1)) - W
    return L, W


def inv_zig_zag(img, N=8):
    vec_im = img.ravel("F")
    zig_zag = zigzag(N)
    blk_out = vec_im[zig_zag.ravel("F")].reshape(img.shape, order="F")
    return blk_out


def zigzag(n):
    """zigzag rows"""

    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)

    xs = range(n)
    tmp = {
        index: n
        for n, index in enumerate(sorted(((x, y) for x in xs for y in xs), key=compare))
    }
    zz_mtx = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            tp = (j, i)
            zz_mtx[j, i] = tmp[tp]
    zz_mtx = zz_mtx.astype(np.int32)
    return zz_mtx


class BitWriter:

    def __init__(self):
        self.sink = io.BytesIO()
        self.buffer = 0
        self.offset = 0

    def write(self, val: int, n: int) -> None:
        """Writes lower `n` bits of `val`."""
        assert 0 <= self.offset <= 32, self.offset
        assert 0 <= n <= 32, n

        room = 32 - self.offset
        if n <= room:
            val &= (1 << n) - 1
            self.buffer |= val << self.offset
            self.offset += n
            return

        self.write(val, room)
        val >>= room
        n -= room

        assert self.offset == 32, self.offset
        self.sink.write(
            int(self.buffer).to_bytes(4, "little", signed=False)
        )  # 4 bytes.
        self.buffer = 0
        self.offset = 0

        assert 0 < n <= 32, n
        self.write(val, n)

    def write_run_length(self, n: int) -> None:
        assert 0 <= n <= 31
        self.write(1 << n, n + 1)

    def finalize(self) -> bytes:
        self.write(1, 1)  # End-of-sequence marker.
        n = (self.offset + 7) // 8
        self.sink.write(int(self.buffer).to_bytes(n, "little", signed=False))
        output = self.sink.getvalue()
        self.sink.close()
        return output


def rle(Input, N=8):
    L = len(Input)
    Output = []
    j = 0
    k = 0
    i = 0

    while i < 2 * L:
        comp = 1
        while j < L:
            if j == L - 1:
                break

            if comp == N:
                break

            if Input[j] == Input[j + 1]:
                comp += 1
            else:
                break
            j += 1

        Output.append(comp)
        Output.append(Input[j])

        if j == L - 1 and Input[j - 1] == Input[j]:
            break

        i += 1
        k += 2
        j += 1

        if j == L:
            if L % 2 == 0:
                Output.append(1)
                Output.append(Input[j - 1])
            else:
                Output.append(1)
                Output.append(Input[j])
            break

    return Output


class BitReader:

    def __init__(self, source: bytes):
        self.source = io.BytesIO(source)
        self.buffer = 0
        self.remain = 0

    def _read_from_source(self) -> None:
        read = self.source.read(4)
        assert read, "Read past the end of the source."
        assert len(read) <= 4, read

        self.buffer = int.from_bytes(read, "little", signed=False)
        self.remain = len(read) * 8

    def read(self, n: int) -> int:
        assert 0 <= n <= 32, n
        assert 0 <= self.remain <= 32, self.remain
        if n <= self.remain:
            val = self.buffer & ((1 << n) - 1)
            self.buffer >>= n
            self.remain -= n
            return val

        val = self.buffer
        offset = self.remain
        n -= self.remain

        self._read_from_source()
        val |= self.read(n) << offset
        return val

    def read_run_length(self) -> int:
        # Maximum is 32.
        if self.buffer != 0:
            n = (self.buffer ^ (self.buffer - 1)).bit_length()
            assert n != 0, n
            assert n <= self.remain, (self.buffer, self.remain)
            self.buffer >>= n
            self.remain -= n
            return n - 1

        n = self.remain
        self._read_from_source()
        return n + self.read_run_length()


def rlgr(x: np.ndarray, L=6) -> bytes:
    """Encodes with Adaptive Run Length Golomb Rice coding.

    Args:
      x: An array of signed integers to be coded.

    Returns:
      A Python `bytes`.
    """
    assert x.dtype == np.int32, x.dtype
    x = np.ravel(x)
    assert np.all(x <= ((1 << 30) - 1))
    assert np.all(-(1 << 30) <= x)

    sink = BitWriter()

    # Constants.
    L = L  # originally this was 4
    U0 = 3
    D0 = 1
    U1 = 2
    D1 = 1
    quotientMax = 24

    # Initialize state.
    k_P = 0
    k_RP = 10 * L

    # Preprocess data from signed to unsigned.
    z = x * 2
    z[z < 0] += 1
    z = np.abs(z)

    N = len(z)
    n = 0
    while n < N:
        k = k_P // L
        k_RP = min(k_RP, 31 * L)
        k_R = k_RP // L

        u = z[n]  # Symbol to encode.

        if k != 0:  # Encode zero runs.
            m = 1 << k  # m = 2**k = expected length of run of zeros

            # Count the run length of zeros, up to m.
            ahead = z[n : n + m]
            zero_count = np.argmax(ahead != 0)  # np.argmax returns the _first_ index.
            if ahead[zero_count] == 0:  # In case (ahead == 0).all() is true.
                zero_count = len(ahead)

            n += zero_count
            if zero_count == len(ahead):
                # Found a complete run of zeros.
                # Write a 0 to denote the run was a complete one.
                sink.write(0, 1)

                # Adapt k.
                k_P += U1
                continue

            # Found a partial run of zeros (length < m).
            # Write a 1 to denote the run was a partial one, and the decoder needs
            # to read k bits to extract the actual run length.
            sink.write(1, 1)
            sink.write(zero_count, k)

            # The next symbol is encoded as z[n] - 1 instead of z[n].
            assert z[n] != 0
            u = z[n] - 1

        # Output GR code for symbol u.
        # bits = bits + gr(u,k_R)
        assert 0 <= u, u
        quotient = u >> k_R  # `quotient` is run-length encoded.
        if quotient < quotientMax:
            sink.write_run_length(quotient)
            sink.write(u, k_R)
        else:
            assert int(u).bit_length() <= 31, (u, u.bit_length())
            sink.write_run_length(quotientMax)
            sink.write(u, 31)

        # Adapt k_R.
        if quotient == 0:
            k_RP = max(0, k_RP - 2)
        elif quotient > 1:
            k_RP += quotient + 1

        # Adapt k.
        if k == 0 and u == 0:
            k_P += U0
        else:  # k > 0 or u > 0
            k_P = max(0, k_P - D0)

        n += 1

    output = sink.finalize()
    return output


def irlgr(source: bytes, N: int) -> np.ndarray:
    """IRLGR decodes bitStream into integers using Adaptive Run Length Golomb Rice.

    Args:
      source: A Python `bytes`.
      N: Number of symbols to decode.

    Returns:
      An array of decoded signed integers.
    """
    # Constants.
    L = 4
    U0 = 3
    D0 = 1
    U1 = 2
    D1 = 1
    quotientMax = 24

    source = BitReader(source)

    # Initialize state.
    k_P = 0
    k_RP = 10 * L

    # Allocate space for decoded unsigned integers.
    output = np.zeros(N, np.int32)

    # Process data one sample at a time (time consuming in Matlab).
    n = 0
    while n < N:
        k = k_P // L
        k_RP = min([k_RP, 31 * L])
        k_R = k_RP // L

        if k != 0:
            is_complete = source.read(1) == 0
            if is_complete:
                zero_count = 1 << k  # 2**k = expected length of run of zeros
                output[n : n + zero_count] = 0
                n += zero_count

                # Adapt k.
                k_P += U1
                continue

            # A partial run was encoded.
            zero_count = source.read(k)
            output[n : n + zero_count] = 0
            n += zero_count

        quotient = source.read_run_length()
        if quotient < quotientMax:
            u = (quotient << k_R) + source.read(k_R)
        else:
            u = source.read(31)
            quotient = u >> k_R

        # Adapt k_R.
        if quotient == 0:
            k_RP = max(0, k_RP - 2)
        elif quotient > 1:
            k_RP += quotient + 1

        # Adapt k.
        if k == 0 and u == 0:
            k_P += U0
        else:  # k > 0 or u > 0
            k_P = max(0, k_P - D0)

        output[n] = u if k == 0 else u + 1
        n += 1

    # Postprocess data from unsigned to signed.
    is_negative = output % 2 == 1
    output = ((output + 1) // 2) * np.where(is_negative, -1, 1)
    return output


def dpcm_smart(tmp):
    fdpcm = np.zeros_like(tmp, dtype=np.int32)
    for j in range(tmp.shape[0]):
        tmp_vec = tmp[j, :]
        tmp_vec = tmp_vec.reshape((len(tmp_vec), 1))
        tmp_vec = tmp_vec.astype(np.int32)
        fdpcm[j, :] = dpcm(tmp_vec, 1).squeeze()
    top = fdpcm[:, 0]
    top = top.reshape((len(top), 1))
    fdpcm[:, 0] = dpcm(top, 1).squeeze()
    return fdpcm


@njit("int32[:, :](int32[:, :], int8)")
def dpcm(x, a):
    #x = x.astype(np.int64)
    m, nx = x.shape
    if hasattr(a, "__len__"):
        p, na = a.shape
    else:
        p = 1
        na = 1
    r = np.zeros((m, nx), dtype=np.int32)
    xtilde = np.zeros((m, nx), dtype=np.int32)
    r[0:p] = np.round(x[0:p])
    xtilde[0:p] = r[0:p]
    for t in range(p, m, 1):
        xhat = np.sum(a*np.flip(xtilde[t-p:t]))
        r[t] = np.round(x[t] - xhat)
        xtilde[t] = r[t] + xhat 
    return r