# MetaHDR: Model-Agnostic Meta-Learning for HDR Image Reconstruction
[![report](https://img.shields.io/badge/arxiv-report-red)](https://tenor.com/bkNeW.gif) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://tenor.com/bkNeW.gif)

| Poster Video                                                                                                | Poster PDF                                                                                                |
|------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| [![PaperVideo](https://img.youtube.com/vi/RvQIZ7aNOc0/0.jpg)](https://www.youtube.com/watch?v=RvQIZ7aNOc0) | [![PosterPDF](./materials/MetaHDR-FlowDiagram.jpg =500x)](https://drive.google.com/file/d/1g1XR3ZRRs6CCAxeiSVdE6fcffyXVxdOh/view?usp=sharing) |

## Getting Started
MetaHDR was implemented and tested on CentOS 7 with python >=3.9.0, and supports GPU and CPU computation. We have also provided a Colab demo for convenience.

First, clone the repository

```
git clone https://github.com/edwin-pan/MetaHDR.git
```

Next, install the requirements

```
pip install requirements.txt
```

## Running the Demo
The demo code provided runs MetaHDR on any LDR image input. 

To run the demo,

```
python3 demo.py --input_folder ./scene_demo --output_folder ./scene_demo/output
```

## Running the Evaluation
Make sure that data is downloaded and formatted correctly (see `data.md`). 

To run evaluation,

```
python3 eval.py --model_dir <PATH-TO-TRAINING-OUTPUTS> --cfg <PATH-TO-CONFIG>
```

## Running the Training
Make sure that data is downloaded and formatted correctly (see `data.md`). Training scripts will log the 

To run training,

```
python3 train.py --cfg <PATH-TO-CONFIG>
```


## Google Colab



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