# Pixel-Wise Motion Deblurring of Thermal Videos

This repository contains code for our RSS 2020 paper. Official proceedings is available
 [here](http://www.roboticsproceedings.org/rss16/p022.pdf) and pre-print is available at 
 [https://arxiv.org/abs/2006.04973](https://arxiv.org/abs/2006.04973).

## Citation
```
@INPROCEEDINGS{Ramanagopal-RSS-20, 
    AUTHOR    = {Manikandasriram Srinivasan Ramanagopal AND Zixu Zhang AND Ram Vasudevan AND Matthew Johnson Roberson}, 
    TITLE     = {{Pixel-Wise Motion Deblurring of Thermal Videos}}, 
    BOOKTITLE = {Proceedings of Robotics: Science and Systems}, 
    YEAR      = {2020}, 
    ADDRESS   = {Corvalis, Oregon, USA}, 
    MONTH     = {July}, 
    DOI       = {10.15607/RSS.2020.XVI.022} 
}
```

## Sample Data

We treat motion deblurring as a per pixel temporal problem. We used 
[FLIR A655sc](https://www.flir.com/products/a655sc/) radiometrically
 calibrated camera to record images at `200Hz` i.e. the sampling period (`5ms`) is roughly
  half the thermal time constant (`~11ms`) of the camera. We are providing sample data of
  an outdoor sequence with the camera panning horizontally as a `.mat` file 
  [here](https://drive.google.com/file/d/14vJBnYxRfuh1kQ462xg5x64KPPYc5oMF/view?usp=sharing)
  
## Requirements  
1. You need IBM cplex optimization studio available [here](https://www.ibm.com/products/ilog-cplex-optimization-studio).
A free academic version is available for students and researchers. In particular, install 
the python API for cplex.

2. Other required python packages:
```
numpy
matplotlib
scikit-image
scipy
h5py
tqdm
```

## Usage

- Clone repository and add to `PYTHONPATH`
```
git clone https://github.com/umautobots/pixelwise-deblurring.git
cd pixelwise-deblurring
export PYTHONPATH=<path/to/pixelwise-deblurring>:$PYTHONPATH
```
- To deblur `<N>` frames starting from `<start_num>`
```
python3.6 ./src/deblur_main.py --matfile <path/to/downloaded/data> --indices <start_num> --N <N> --output-prefix <path/to/output/folder>
```
You can provide comma separated list of starting indices and output files are automatically named as `{output_prefix}_{start_num}_{start_num+N}.npz`

Note:
Since each pixel is independently processed, `~80k` optimization problems needs to be solved which is currently slow.
The code will automatically use the maximum number of CPU cores available for parallel processing.

- To view the processed files, use:
```
python3.6 ./src/view_processed_frames.py --filename <path/to/npz/file>
```

 