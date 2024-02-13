This is a simple implementation of AVC in Python.

It is intended for prototyping and is far from a functional AVC implementation.

The code is structured so that it can be called from a parent directory, that is:

```
python -m STAC_AVC.test_stac_avc 
```

Most of the functions inside hessian_compute/ require JAX. However, the AVC code does not depend on any of them, and it can be run independently.

There are two test functions: test_stac_avc.py, which can be run without JAX, and test_stac_avc_jax.py, which depends on JAX.

The LPIPS implementation in JAX can be found in https://github.com/wilson1yan/lpips-jax.

# TODO

- [ ] Add support for perceptual metrics such as MS-SSIM, LPIPS, and BRISQUE
- [ ] Decouple AVC and Q-AVC code totally
- [ ] Add Chroma support
- [ ] Add 8x8 transforms
- [ ] Add comments so that it is easier to interpret
- [ ] dec_cavlc is not working properly
