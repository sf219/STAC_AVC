This is a straightforward implementation of AVC in Python.

It is intended for prototyping and is far from a functional AVC implementation.

The code is structured so that it can be called from a parent directory, that is:

```
python -m STAC_AVC.test_stac_avc 
```

# TODO

- [ ] Add support for perceptual metrics such as MS-SSIM, LPIPS, and BRISQUE
- [ ] Decouple AVC and Q-AVC code totally
- [ ] Add Chroma support
- [ ] Add 8x8 transforms
- [ ] Add comments so that it is easier to interpret
- [ ] dec_cavlc is not working properly
