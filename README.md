# AI-IMU Dead-Reckoning [[IEEE paper](https://ieeexplore.ieee.org/document/9035481), [ArXiv paper](https://arxiv.org/pdf/1904.06064.pdf)]

_1.10%_ translational error on the [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) odometry sequences with __only__ an Inertial Measurement Unit.

![Results on sequence 08](temp/08.gif)

## Overview

In the context of intelligent vehicles, robust and accurate dead reckoning based on the Inertial Measurement Unit (IMU) may prove useful to correlate feeds from imaging sensors, to safely navigate through obstructions, or for safe emergency stop in the extreme case of other sensors failure.

This repo contains the code of our novel accurate method for dead reckoning of wheeled vehicles based only on an IMU. The key components of the method are the Kalman filter and the use of deep neural networks to dynamically adapt the noise parameters of the filter. Our dead reckoning inertial method based only on the IMU accurately estimates 3D position, velocity, orientation of the vehicle and self-calibrates the IMU biases. We achieve on the KITTI odometry dataset on average a 1.10% translational error and the algorithm competes with top-ranked methods which, by contrast, use LiDAR or stereo vision.

![Structure of the approach](temp/structure.jpg)

The above figure illustrates the approach which consists of two main blocks summarized as follows:
1. the filter integrates the inertial measurements with exploits zero lateral and vertical velocity as measurements with covariance matrix to refine its estimates, see the figure below;
2. the noise parameter adapter determines in real-time the most suitable covariance noise matrix. This deep learning based adapter converts directly raw IMU signals into covariance matrices without requiring knowledge of any state estimate nor any other quantity.


![Structure of the filter](temp/iekf.jpg)

## Code
Our implementation is done in Python. We use [Pytorch](https://pytorch.org/) for the adapter block of the system. The code was tested under Python 3.5.
 
### Installation & Prerequies
1.  Install [pytorch](http://pytorch.org). We perform all training and testing on its development branch.
    
2.  Install the following required packages, `matplotlib`, `numpy`, `termcolor`, `scipy`, `navpy`, e.g. with the pip3 command
```
pip3 install matplotlib numpy termcolor scipy navpy
```
    
4.  Clone this repo
```
git clone https://github.com/mbrossar/ai-imu-dr.git
```

### Testing
1. Download reformated pickle format of the 00-11 KITTI IMU raw data at this [url](https://www.dropbox.com/s/ey41xsvfqca30vv/data.zip), extract and copy then in the `data` folder.
```
wget "https://www.dropbox.com/s/ey41xsvfqca30vv/data.zip"
mkdir ai-imu-dr/results
unzip data.zip -d ai-imu-dr
rm data.zip
```
These file can alternatively be generated after download the KITTI raw data and setting `read_data = 1` in the `main.py` file.

2. Download training parameters at this [url](https://www.dropbox.com/s/77kq4s7ziyvsrmi/temp.zip), extract and copy in the `temp` folder.
```
wget "https://www.dropbox.com/s/77kq4s7ziyvsrmi/temp.zip"
unzip temp.zip -d ai-imu-dr/temp
rm temp.zip
```
4. Test the filters !
```
cd ai-imu-dr/src
python3 main_kitti.py
```
This first launches the filters for the all sequences. Then, results are plotted. Note that the parameters are  trained on sequences 00, 01, 04-11, so only sequence 02 is a test sequence.

### Training
You can train for testing another sequence (we do not find difference in the results) or for our own sequence by modifying the dataset class.


## Paper
The paper M. Brossard, A. Barrau and S. Bonnabel, "AI-IMU Dead-Reckoning," in _IEEE Transactions on Intelligent Vehicles_, 2020, relative to this repo is available at this [url](https://cloud.mines-paristech.fr/index.php/s/8YDqD0Y1e6BWzCG).


### Citation

If you use this code in your research, please cite:

```
@article{brossard2019aiimu,
  author = {Martin Brossard and Axel Barrau and Silv\`ere Bonnabel},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title = {{AI-IMU Dead-Reckoning}},
  year = {2020}
}
```

### Authors
Martin Brossard*, Axel Barrau° and Silvère Bonnabel*

*MINES ParisTech, PSL Research University, Centre for Robotics, 60 Boulevard Saint-Michel, 75006 Paris, France

°Safran Tech, Groupe Safran, Rue des Jeunes Bois-Châteaufort, 78772, Magny Les Hameaux Cedex, France
