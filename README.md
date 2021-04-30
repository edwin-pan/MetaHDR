# MetaHDR: Model-Agnostic Meta-Learning for HDR Image Reconstruction
[![report](https://img.shields.io/badge/arxiv-report-red)](https://arxiv.org/abs/2103.12545) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WeyrYHh2BvRraUz9Xorr99Iy8CGM6_aM?usp=sharing)

| Poster Video                                                                                                | Poster PDF                                                                                                |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [![PaperVideo](https://img.youtube.com/vi/RvQIZ7aNOc0/0.jpg)](https://www.youtube.com/watch?v=RvQIZ7aNOc0) | [![PosterPDF](./materials/Mini-Meta-Figure.png)](https://drive.google.com/file/d/1g1XR3ZRRs6CCAxeiSVdE6fcffyXVxdOh/view?usp=sharing) |

## Directory Structure
    .
    ├── configs                 # Stores yaml files for configuring training, evaluation, demo, etc.
    ├── data                    # Directory for data. See data.md for details on getting data.
    ├── materials               # Miscellaneous assets for the report, poster, video, etc.
    ├── scene_demo              # Small sample of input data used to demo adaptive forward-pass.
    ├── src                     
        ├── core                # Contains scripts used for configuration as well as the loss metrics used.
        ├── dataset             # Contains scripts used to read in the data, a data generator which will sample batches of data, as well as an hdr visualization script.
        └── model               # Contains scripts which define the UNet for the inner model, as well as functions for training, validating, and evaluating our model with MAML.
    ├── .gitignore
    ├── README.md
    ├── data.md
    ├── demo.py                 # Runs an adaptive forward pass to demonstrate deployed usage.
    ├── eval.py                 # Runs an evaluation of a trained model.
    ├── requirements.txt        # Required dependencies to run this model.
    └── train.py                # Runs training iterations.

### Author Contact:
Edwin Pan <edwinpan@stanford.edu>

Anthony Vento <avento@stanford.edu>

## Getting Started
MetaHDR was implemented and tested on CentOS 7 with python >=3.9.0, and supports GPU and CPU computation. We have also provided a Colab demo for convenience.

First, clone the repository

```
git clone https://github.com/edwin-pan/MetaHDR.git
```

Next, install the requirements

```
pip install -r requirements.txt
```

## Running the Demo
The demo code provided runs MetaHDR on any LDR image input. We have provided a sample scene input directory at `./scene_demo`. LDR images need to be placed in `./scene_demo/LDR/` and the corresponding HDR labels should go to `./scene_demo/HDR/`. 

To run the demo,

```
python3 demo.py --input_folder ./scene_demo --output_folder ./scene_demo/output --cfg configs/config_demo.yaml
```

A new directory at `./scene_demo/output/` will contain the output HDR image.


## Running the Evaluation
Make sure that data is downloaded and formatted correctly (see [data.md](data.md)). 

To run evaluation,

```
python3 eval.py --model_dir <PATH-TO-TRAINING-OUTPUTS> --cfg <PATH-TO-CONFIG>
```

## Running the Training
Make sure that data is downloaded and formatted correctly (see [data.md](data.md)). Training scripts will log the results in the output directory specified by the configuration file.

To run training,

```
python3 train.py --cfg <PATH-TO-CONFIG>
```


## Google Colab
To quickly get a sense of how MetaHDR performs, we have provided a Google Colab notebook version of the demo that runs out of the box. The demo runs on the cloud for free.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WeyrYHh2BvRraUz9Xorr99Iy8CGM6_aM?usp=sharing)


## Acknowledgement
This work was completed as a Final Project for EE 367 / CS 448I: Computational Imaging and Display at Stanford University. We would like to thank our professor, Dr. Gordon Wetzstein, for his valuable instruction throughout the quarter and our project mentor, Cindy Nguyen, for giving us insights into our project.


## Citation
```bibtex
@inproceedings{metahdr2021,
  title={MetaHDR: Model-Agnostic Meta-Learning for HDR Image Reconstruction},
  author={Pan, Edwin and Vento, Anthony},
  month={March},
  year={2021}
}
```

## References
A full list of references for this project can be found in the arxiv paper. 