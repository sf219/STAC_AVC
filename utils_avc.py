# Standard libraries
import numpy as np
import scipy.io

y_table = np.array(
    [[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60,
                                        55], [14, 13, 16, 24, 40, 57, 69, 56],
     [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103,
                                        77], [24, 35, 55, 64, 81, 104, 113, 92],
     [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]],
    dtype=np.float32)

#
c_table = np.empty((8, 8), dtype=np.float32)
c_table.fill(99)
c_table[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66],
                            [24, 26, 56, 99], [47, 66, 99, 99]]).T

def quality_to_factor(quality):
    """ Calculate factor corresponding to quality
    Input:
        quality(float): Quality for jpeg compression
    Output:
        factor(float): Compression factor
    """
    if quality < 50:
        factor = 50. / quality
    else:
        factor = 2 - quality/50
    delta = factor*y_table[0, 0]
    return factor, delta


def enc_cavlc(data, nL: int, nU: int):
    mat_file = scipy.io.loadmat('video_codecs/data/table.mat')
    Table_coeff0 = mat_file['Table_coeff0']
    Table_coeff1 = mat_file['Table_coeff1']
    Table_coeff2 = mat_file['Table_coeff2']
    Table_coeff3 = mat_file['Table_coeff3']
    Table_run = mat_file['Table_run']
    Table_zeros = mat_file['Table_zeros']

    bits = ""

    # Convert 4x4 matrix data into a 1x16 data of zig-zag scan
    l = []
    scan = [(1, 1), (1, 2), (2, 1), (3, 1), (2, 2), (1, 3), (1, 4), (2, 3), (3, 2), (4, 1), (4, 2), (3, 3), (2, 4), (3, 4), (4, 3), (4, 4)]

    for m, n in scan:
        l.append(data[m - 1][n - 1])

    i_last = 16
    # find the last non-zero co-eff in reverse order
    while i_last > 0 and l[i_last - 1] == 0:
        i_last -= 1

    i_total = 0  # Total non-zero coefficients
    i_total_zero = 0  # Total zeros
    i_trailing = 0
    sign = ""  # find sign for trailing ones
    idx = 0
    run = np.zeros(0)
    level = np.zeros(0)

    # find level, trailing ones(sign), run, and total zero values
    while i_last > 0 and abs(l[i_last - 1]) == 1 and i_trailing < 3:
        level = np.append(level, l[i_last - 1])
        i_total += 1
        i_trailing += 1

        if l[i_last - 1] == -1:
            sign = sign + "1"
        else:
            sign = sign + "0"

        run_tmp = 0
        i_last -= 1
        while i_last > 0 and l[i_last - 1] == 0:
            run_tmp += 1
            i_total_zero += 1
            i_last -= 1
        run = np.append(run, run_tmp)
        idx += 1

    while i_last > 0:
        level = np.append(level, l[i_last - 1])
        i_total += 1

        run_tmp = 0
        i_last -= 1
        while i_last > 0 and l[i_last - 1] == 0:
            run_tmp += 1
            i_total_zero += 1
            i_last -= 1
        run = np.append(run, run_tmp)
        idx += 1

    n = (nL + nU + 1) >> 1
    #if nL > 0 and nU > 0:
    #    n = (nL + nU + 1).astype(int) >> 1
    #else:
    #    n = nL + nU

    # Write coeff_token
    if 0 <= n < 2:
        Table_coeff = Table_coeff0
    elif 2 <= n < 4:
        Table_coeff = Table_coeff1
    elif 4 <= n < 8:
        Table_coeff = Table_coeff2
    elif n >= 8:
        Table_coeff = Table_coeff3

    coeff_token = Table_coeff[i_total][i_trailing]
    bits += str(coeff_token[0])

    if i_total == 0:
        return bits

    if i_trailing > 0:
        bits += sign

    # Encode the levels of remaining non-zero coefficients
    if i_total > 10 and i_trailing < 3:
        i_sufx_len = 1
    else:
        i_sufx_len = 0
    for i in range(i_trailing, i_total):
        if level[i] < 0:
            i_level_code = -2 * level[i] - 1
        else:
            i_level_code = 2 * level[i] - 2

        if i == i_trailing and i_trailing < 3:
            i_level_code -= 2

        if (i_level_code.astype(int) >> i_sufx_len) < 14:
            level_prfx = i_level_code.astype(int) >> i_sufx_len
            while level_prfx > 0:
                bits += "0"
                level_prfx -= 1
            bits += "1"

            if i_sufx_len > 0:
                tmp_bin = bin(i_level_code.astype(int))[2:]
                level_sufx = tmp_bin[-i_sufx_len:].zfill(i_sufx_len)
                x = len(level_sufx)
                if (x > i_sufx_len):
                    level_sufx = level_sufx[x - i_sufx_len:x]        
                bits += level_sufx

        elif i_sufx_len == 0 and i_level_code < 30:
            level_prfx = 14
            while level_prfx > 0:
                bits += "0"
                level_prfx -= 1
            bits += "1"
            tmp_bin = bin(i_level_code.astype(int) - 14)[2:]
            level_sufx = tmp_bin[-4:].zfill(4)
            bits += level_sufx

        elif i_sufx_len > 0 and (i_level_code.astype(int) >> i_sufx_len) == 14:
            level_prfx = 14
            while level_prfx > 0:
                bits += "0"
                level_prfx -= 1
            bits += "1"
            tmp_bin = bin(i_level_code.astype(int))[2:]
            level_sufx = tmp_bin[-i_sufx_len:].zfill(i_sufx_len)            
            bits += level_sufx

        else:
            level_prfx = 15
            while level_prfx > 0:
                bits += "0"
                level_prfx -= 1
            bits += "1"

            i_level_code -= 2 ** i_sufx_len

            if i_sufx_len == 0:
                i_level_code -= 15

            if i_level_code >= 2 ** 12 or i_level_code < 0:
                print("Overflow occurred")

            tmp_bin = bin(i_level_code.astype(int))[2:]
            level_sufx = tmp_bin[-12:].zfill(12)
            x = len(level_sufx)
            if (x > 12):
                level_sufx = level_sufx[x - 12:x]
            bits += level_sufx

        if i_sufx_len == 0:
            i_sufx_len += 1
        thres = 3 << i_sufx_len-1
        if abs(level[i]) > thres and i_sufx_len < 6:
            i_sufx_len += 1

    # Encode Total zeros
    if i_total < 16:
        total_zeros = Table_zeros[i_total-1][i_total_zero]
        bits += str(total_zeros[0])
    # Encode each run of zeros
    i_zero_left = i_total_zero
    if i_zero_left >= 1:
        for i in range(i_total):
            if i_zero_left > 0 and (i+1) == i_total:
                break
            if i_zero_left >= 1:
                i_zl = int(min(i_zero_left, 7))
                run_before = Table_run[run[i].astype(int)][i_zl-1]
                bits += str(run_before[0])
                i_zero_left -= run[i]
    return bits


def dec_cavlc(bits, nL, nU):
    # TODO: This is not working
    # Load the table containing all the tables
    mat_file = scipy.io.loadmat('video_codecs/data/table.mat')
    Table_coeff0 = mat_file['Table_coeff0']
    Table_coeff1 = mat_file['Table_coeff1']
    Table_coeff2 = mat_file['Table_coeff2']
    Table_coeff3 = mat_file['Table_coeff3']
    Table_run = mat_file['Table_run']
    Table_zeros = mat_file['Table_zeros']

    # find n parameter (context adaptive)
    n = (nL + nU + 1) >> 1

    """
    if (nL > 0) and (nU > 0):
        n = (nL + nU) / 2
    elif (nL > 0) or (nU > 0):
        n = nL + nU
    else:
        n = 0
    """
    # Coeff_token mapping
    # Rows are the total coefficient(0-16) and columns are the trailing ones(0-3)
    # TABLE_COEFF0,1,2,3 ARE STORED IN TABLE.MAT OR CAVLC_TABLES.M FILE
    # Choose proper Table_coeff based on n value
    if 0 <= n < 2:
        Table_coeff = Table_coeff0
    elif 2 <= n < 4:
        Table_coeff = Table_coeff1
    elif 4 <= n < 8:
        Table_coeff = Table_coeff2
    elif n >= 8:
        Table_coeff = Table_coeff3

    i = 0
    coeff_token = ''

    # Find total coefficients and trailing ones
    while i < len(bits):
        coeff_token += bits[i]
        x = Table_coeff == coeff_token
        r, c = np.where(x)
        if r.size > 0 and c.size > 0:
            break
        i += 1

    # Find total coefficients and trailing ones
    i_total = r[0] - 1
    i_trailing = c[0] - 1

    # if no coefficients return 4x4 empty blocks of data
    if i_total == 0:
        data = np.zeros((4, 4))
        return data, i

    k = 0
    m = i_trailing
    level = np.zeros(i_total)

    while m > 0:
        if bits[i] == '0':
            level[k] = 1
        elif bits[i] == '1':
            level[k] = -1
        k += 1
        m -= 1
        i += 1

    # Decode the non-zero coefficient/level values
    if (i_total > 10) and (i_trailing < 3):
        i_sufx_len = 1
    else:
        i_sufx_len = 0

    while k < i_total:
        # Decode level prefix
        level_prfx, i = dec_prfx(bits, i)

        # Decode level suffix
        level_sufx_size = 0

        if (i_sufx_len > 0) or (level_prfx >= 14):
            if (level_prfx == 14) and (i_sufx_len == 0):
                level_sufx_size = 4
            elif level_prfx >= 15:
                level_sufx_size = level_prfx - 3
            else:
                level_sufx_size = i_sufx_len

        if level_sufx_size == 0:
            level_sufx = 0
        else:
            sufx = bits[i: i + level_sufx_size]
            level_sufx = int(sufx, 2)
            i += level_sufx_size

        i_level_code = (min(15, level_prfx) << i_sufx_len) + level_sufx

        if (level_prfx >= 15) and (i_sufx_len == 0):
            i_level_code += 15
        if level_prfx >= 16:
            i_level_code += (1 << (level_prfx - 3)) - 4096

        if (k == i_trailing + 1) and (i_trailing < 3):
            i_level_code += 2

        if i_level_code % 2 == 0:  # i_level_code is even
            level[k] = (i_level_code + 2) >> 1
        else:  # odd number
            level[k] = (-i_level_code - 1) >> 1

        if i_sufx_len == 0:
            i_sufx_len = 1

        if (abs(level[k]) > (3 << (i_sufx_len - 1))) and (i_sufx_len < 6):
            i_sufx_len += 1

        k += 1

    # Decode total zeros
    s = ''
    i_total_zero = 0

    if i_total == 16:
        i_zero_left = 0
    else:
        while i < len(bits):
            s += bits[i]
            x = Table_zeros[i_total - 1] == s
            r = np.where(x)[0]
            if r.size > 0:
                i_total_zero = r[0] - 1
                break
            i += 1

    # Decode run information
    i_zero_left = i_total_zero
    j = 0
    ss = ''
    run = np.zeros(len(level))

    while i_zero_left > 0:
        while (j < i_total) and (i_zero_left > 0) and (i < len(bits)):
            ss += bits[i]
            i_zl = int(min(i_zero_left, 7))
            x = Table_run[int(run[j])][i_zl - 1] == ss
            r = np.where(x)[0]
            if r.size > 0:
                run[j] = r[0] - 1
                i_zero_left -= run[j]
                j += 1
                ss = ''
            i += 1
        if i_zero_left > 0:
            run[j] = i_zero_left
            i_zero_left = 0

    # Combine level and run information
    k = i_total + i_total_zero
    l = np.zeros(16)

    while k > 0:
        for j in range(len(level)):
            l[k - 1] = level[j]
            k -= 1
            k -= int(run[j])

    # Reorder the data into 4x4 block
    scan = np.array([[1, 1], [1, 2], [2, 1], [3, 1], [2, 2], [1, 3], [1, 4], [2, 3],
                     [3, 2], [4, 1], [4, 2], [3, 3], [2, 4], [3, 4], [4, 3], [4, 4]])

    data = np.zeros((4, 4))

    for k in range(16):
        m, n = scan[k, 0], scan[k, 1]
        data[m - 1, n - 1] = l[k]  # l contains the reordered data

    return data, i

def dec_prfx(bits, i):
    level_prfx = 0
    while bits[i] == '0':
        level_prfx += 1
        i += 1
    level_prfx += 1
    return level_prfx, i
