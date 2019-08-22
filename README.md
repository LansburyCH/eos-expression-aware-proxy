# eos-expression-aware-proxy
This repo holds the Expression-Aware Proxy Generation part of the paper *Photo-Realistic Facial Details Synthesis From Single Image*, ICCV 2019. It is modified based on [patrikhuber](https://github.com/patrikhuber)'s fantastic work [eos](https://github.com/patrikhuber/eos) and enables extra options:

 - input initial shape/expression coefficients
 - whether to fix shape/expression coefficients during fitting
 - output various information (depth map, normal map, etc.) that may be useful for further processing

Note: This repo is not kept in pace with changes in [eos](https://github.com/patrikhuber/eos).

## Getting Started

### BFM2017
Download [BFM2017](https://faces.dmi.unibas.ch/bfm/bfm2017.html) and copy `model2017-1_bfm_nomouth.h5` to `bfm2017/`.

Install an older version of `eos-py` by
```
pip install --force-reinstall eos-py==0.16.1
```

Run `share/scripts/convert-bfm2017-to-eos.py` to generate `bfm2017-1_bfm_nomouth.bin` in `bfm2017/`.

### Prerequisites

 - Visual Studio 2017 (>=15.5)
 - Boost (>=1.50.0)
 - OpenCV (>=2.4.3)
 - Ceres (if compiling `fit-model-ceres`)

### Installing
Please refer to [eos](https://github.com/patrikhuber/eos) for detailed instructions. For Windows users, It is recommended to use [vcpkg](https://github.com/Microsoft/vcpkg/) to install dependencies. In such case, edit `D:/repo/vcpkg/scripts/buildsystems/vcpkg.cmake` in `CMakeLists.txt` to appropriate path.

### Usage
To check whether installation is successful, go to `CMAKE_INSTALL_PREFIX/bin/`  and run (may need to first copy related .dll to this directory)
```
./fit-model.exe
```
A couple of output files should be generated in `CMAKE_INSTALL_PREFIX/bin/data/` where `image_0010.out.obj` is the face mesh. 

To run on specified image, you need to first obtain facial landmarks (Multi-PIE 68 points style) for the image. Then use `-i` to specify path to the image and `-l` the path to the landmark file. Please refer to all available options through `-h`.

To use expression prior as in our paper, save initial expression coefficients to a `.txt` with one coefficient per line. When running `fit-model`, specify with extra flags

`--init-expression-coeffs-fp PATH_TO_TXT --fix-expression-coeffs 1`

## Additional Output
For `fit-model`, by specifying `--save-mode all`, it will generate many additional output (saved as .dat files). Below are descriptions for some of them:

 - *.depthbuffer.dat - depth of each pixel
 - *.coordsbuffer.dat - 3D position of each pixel
 - *.texcoordsbuffer.dat - texture coordinate of each pixel
 - *.remap_dst_to_src.dat - image coordinate of each texel
 - *.affine_camera_matrix_with_z.dat - full transformation and projection matrix
 - *.model_view_matrix.dat - transformation matrix
 - *.src_normal.dat - normal of each pixel
 - *.dst_normal.dat - normal of each texel
 - *.face_normal.dat - normal of each face
 - *.src_pixel_face.dat - face id of each pixel
 - *.dst_pixel_face.dat - face id of each texel

Matlab functions for read/write these .dat files are available in matlab/. The nan value is set as a very large number, which can be removed by `xxx(xxx == max(xxx(:))) = nan`.

## Citation
If using this code in your own work, please cite the following paper along with the publication associated with [eos](https://github.com/patrikhuber/eos).
```
@article{chen2019photo,
  title     = {Photo-Realistic Facial Details Synthesis from Single Image},
  author    = {Anpei Chen, Zhang Chen, Guli Zhang, Ziheng Zhang, Kenny Mitchell and Jingyi Yu},
  journal   = {arXiv preprint arXiv:1903.10873},
  year      = {2019}
}
```

