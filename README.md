# FAsT-Match

Python implementation of FAsT-Match.

Note: horrible performance compared to [C++ version](https://github.com/vinhtq115/FAsT-Match) due to for loops not parallelized. May produce inaccurate randomly.

## Required packages
* `numpy>=1.21`
* `opencv-python>=4.5`

## Usage

```
usage: main.py [-h] [-o OUTPUT] image_file template_file

FAsT-Match

positional arguments:
  image_file            Input image file
  template_file         Template image file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Path to output image
```