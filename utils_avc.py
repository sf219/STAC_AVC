# Standard libraries
import numpy as np
import scipy.io
from PIL import Image, ImageOps
import math
from scipy.fftpack import dct, idct

mat_file = scipy.io.loadmat('STAC_AVC/data/table.mat')
Table_coeff0 = mat_file['Table_coeff0']
Table_coeff1 = mat_file['Table_coeff1']
Table_coeff2 = mat_file['Table_coeff2']
Table_coeff3 = mat_file['Table_coeff3']
Table_run = mat_file['Table_run']
Table_zeros = mat_file['Table_zeros']

def enc_cavlc(data, nL: int, nU: int):

    bits = ""

    # Define the scan order
    scan = np.array([(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2), (0, 3), (1, 2), (2, 1), (3, 0), (3, 1), (2, 2), (1, 3), (2, 3), (3, 2), (3, 3)])

    # Use advanced indexing to get the elements in zig-zag order
    l = data[scan[:, 0], scan[:, 1]]

    """
    # Convert 4x4 matrix data into a 1x16 data of zig-zag scan
    l = []
    scan = [(1, 1), (1, 2), (2, 1), (3, 1), (2, 2), (1, 3), (1, 4), (2, 3), (3, 2), (4, 1), (4, 2), (3, 3), (2, 4), (3, 4), (4, 3), (4, 4)]

    for m, n in scan:
        l.append(data[m - 1][n - 1])
    """
    
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


def bdsnr(metric_set1, metric_set2):
  """
  BJONTEGAARD    Bjontegaard metric calculation
  Bjontegaard's metric allows to compute the average gain in psnr between two
  rate-distortion curves [1].
  rate1,psnr1 - RD points for curve 1
  rate2,psnr2 - RD points for curve 2
  returns the calculated Bjontegaard metric 'dsnr'
  code adapted from code written by : (c) 2010 Giuseppe Valenzise
  http://www.mathworks.com/matlabcentral/fileexchange/27798-bjontegaard-metric/content/bjontegaard.m
  """
  # pylint: disable=too-many-locals
  # numpy seems to do tricks with its exports.
  # pylint: disable=no-member
  # map() is recommended against.
  # pylint: disable=bad-builtin
  rate1 = [x[0] for x in metric_set1]
  psnr1 = [x[1] for x in metric_set1]
  rate2 = [x[0] for x in metric_set2]
  psnr2 = [x[1] for x in metric_set2]

  log_rate1 = map(math.log, rate1)
  log_rate2 = map(math.log, rate2)

  # Best cubic poly fit for graph represented by log_ratex, psrn_x.
  poly1 = np.polyfit(log_rate1, psnr1, 3)
  poly2 = np.polyfit(log_rate2, psnr2, 3)

  # Integration interval.
  min_int = max([min(log_rate1), min(log_rate2)])
  max_int = min([max(log_rate1), max(log_rate2)])

  # Integrate poly1, and poly2.
  p_int1 = np.polyint(poly1)
  p_int2 = np.polyint(poly2)

  # Calculate the integrated value over the interval we care about.
  int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
  int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

  # Calculate the average improvement.
  if max_int != min_int:
    avg_diff = (int2 - int1) / (max_int - min_int)
  else:
    avg_diff = 0.0
  return avg_diff


def bdrate(metric_set1, metric_set2):
    """
    BJONTEGAARD    Bjontegaard metric calculation
    Bjontegaard's metric allows to compute the average % saving in bitrate
    between two rate-distortion curves [1].
    rate1,psnr1 - RD points for curve 1
    rate2,psnr2 - RD points for curve 2
    adapted from code from: (c) 2010 Giuseppe Valenzise
    """
    # numpy plays games with its exported functions.
    # pylint: disable=no-member
    # pylint: disable=too-many-locals
    # pylint: disable=bad-builtin
    rate1 = [float(x[0]) for x in metric_set1]
    psnr1 = [float(x[1]) for x in metric_set1]
    rate2 = [float(x[0]) for x in metric_set2]
    psnr2 = [float(x[1]) for x in metric_set2]

    #breakpoint()

    log_rate1 = np.log(np.array(rate1))
    log_rate2 = np.log(np.array(rate2))
    #log_rate1 = map(math.log, rate1)
    #log_rate2 = map(math.log, rate2)

  # Best cubic poly fit for graph represented by log_ratex, psrn_x.
    try:
        poly1 = np.polyfit(psnr1, log_rate1, 3)
        poly2 = np.polyfit(psnr2, log_rate2, 3)
    except:
        return 100

    # Integration interval.
    min_int = max([min(psnr1), min(psnr2)])
    max_int = min([max(psnr1), max(psnr2)])

    # find integral
    p_int1 = np.polyint(poly1)
    p_int2 = np.polyint(poly2)

    # Calculate the integrated value over the interval we care about.
    int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
    int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

    # Calculate the average improvement.
    avg_exp_diff = (int2 - int1) / (max_int - min_int)

    # In really bad formed data the exponent can grow too large.
    # clamp it.
    if avg_exp_diff > 200:
        avg_exp_diff = 200

    # Convert to a percentage.
    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return avg_diff

def read_image_resize_rect(img_path, im_size=(256, 256)):
    im = Image.open(img_path)
    new_height = im_size[0]
    new_width = im_size[1]
    resized_image = ImageOps.fit(im, (new_width, new_height), Image.BICUBIC)
    resized_image = np.array(resized_image).astype(np.float64)
    if (len(resized_image.shape) == 3):
        resized_image = rgb2ycbcr(resized_image/255)
        depth = resized_image.shape[2]
    else:
        depth = 1    
    resized_image = np.round(resized_image)
    return resized_image, depth

def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def rgb2ycbcr(rgb):
    # This matrix comes from a formula in Poynton's, "Introduction to
    # Digital Video" (p. 176, equations 9.6).
    # T is from equation 9.6: ycbcr = origT * rgb + origOffset;
    origT = np.array([[65.481, 128.553, 24.966],
                      [-37.797, -74.203, 112],
                      [112, -93.786, -18.214]])
    origOffset = np.array([16, 128, 128])

    # Initialize variables
    isColormap = False

    # Must reshape colormap to be m x n x 3 for transformation
    if rgb.ndim == 2:
        # colormap
        isColormap = True
        colors = rgb.shape[0]
        rgb = rgb.reshape((colors, 1, 3))

    # The formula ycbcr = origT * rgb + origOffset, converts a RGB image in the
    # range [0 1] to a YCbCr image where Y is in the range [16 235], and Cb and
    # Cr are in that range [16 240]. For each class type, we must calculate
    # scaling factors for origT and origOffset so that the input image is
    # scaled between 0 and 1, and so that the output image is in the range of
    # the respective class type.
    if np.issubdtype(rgb.dtype, np.integer):
        if rgb.dtype == np.uint8:
            scaleFactorT = 1/255
            scaleFactorOffset = 1
        elif rgb.dtype == np.uint16:
            scaleFactorT = 257/65535
            scaleFactorOffset = 257
    else:
        scaleFactorT = 1
        scaleFactorOffset = 1

    # The formula ycbcr = origT*rgb + origOffset is rewritten as
    # ycbcr = scaleFactorForT * origT * rgb + scaleFactorForOffset*origOffset.
    # To use np.einsum, we rewrite the formula as ycbcr = T * rgb + offset,
    # where T and offset are defined below.
    T = scaleFactorT * origT
    offset = scaleFactorOffset * origOffset
    ycbcr = np.zeros_like(rgb)
    for p in range(3):
        ycbcr[:, :, p] = T[p, 0]*rgb[:, :, 0] + T[p, 1]*rgb[:, :, 1] + T[p, 2]*rgb[:, :, 2] + offset[p]   

    if isColormap:
        ycbcr = ycbcr.squeeze()
    return ycbcr

def dct_2d(a):
    return dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct_2d(a):
    return idct( idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')
