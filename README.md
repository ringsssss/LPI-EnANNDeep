# LPI-EnANNDeep
EnANNDeep: An Ensemble-based lncRNA-protein Interaction Prediction Framework with Adaptive k-Nearest Neighbor Classifier and Deep Models

## Data
Data is available at [NONCODE](http://www.noncode.org/), [NPInter](http://bigdata.ibp.ac.cn/npinter3/index.htm), and [PlncRNADB](http://bis.zju.edu.cn/PlncRNADB/).

## Environment
Install python3 for running this code. And these packages should be satisfied:
* tensorflow == 1.14.0
* keras == 2.3.0
* pandas == 1.1.5
* numpy == 1.19.3
* scikit-learn == 0.24.2
* deep-forest == 0.1.5

## Usage
To run the model, default 5 fold cross validation
```
python examples/network.py
```
