# Skin Lesions Classification with Deep Convolutional Neural Network

This is a 40-hour project for CIS 5526 Machine Learning. For full description and analysis please refer to Project_Report.pdf. 

Future work in better training strategy and exploring other models such as Xception and creating a bigger ensemble can help the model performs better than this!

## Files Description

* Final report: Project_Report.pdf

* Exploratory data analysis: Skin_Cancer_EDA.ipynb

* Baseline model: Baseline_CNN.ipynb

* Fine-tuning the last convolutional block of VGG16: Fine_Tuning_VGG16.ipynb

* Fine-tuning the top 2 inception blocks of InceptionV3: Fine_Tuning_InceptionV3.ipynb

* Fine-tuning the Inception-ResNet-C of Inception-ResNet V2: Fine_Tuning_InceptionResNet.ipynb

* Fine-tuning the last dense block of DenseNet 201: Fine_Tuning_DenseNet.ipynb

* Fine-tuning all layers of pretrained Inception V3 on ImageNet: Retraining_InceptionV3.ipynb

* Fine-tuning all layers of pretrained DenseNet 201 on ImageNet: Retraining_DenseNet.ipynb

* Ensemble model of the fully fine-tuned Inception V3 and DenseNet 201 (best result): Ensemble_Models.ipynb 


## Technical Issue

I'm using Keras 2.2.4 and Tensorflow 1.11. Batch-Norm layer in this version of Keras is implemented in a way that: during training your network will always use the mini-batch statistics either the BN layer is frozen or not; also during inference you will use the previously learned statistics of the frozen BN layers. As a result, if you fine-tune the top layers, their weights will be adjusted to the mean/variance of the new dataset. Nevertheless, during inference they will receive data which are scaled differently because the mean/variance of the original dataset will be used. Consequently, if use Keras's example codes for fine-tuning Inception V3 or any network with batch norm layer, the results will be very bad. Please refer to issue [#9965](https://github.com/keras-team/keras/pull/9965) and [#9214](https://github.com/keras-team/keras/issues/9214). One temporary solution is: 

```python
for layer in pre_trained_model.layers:
    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False
```

## Results

| Models        | Validation           | Test            |  Depth          | # Params          |
| ------------- |:-------------:| :-------------:| :-------------:| :-------------:| 
|   Baseline   | 77.48% |76.54% | 11 layers | 2,124,839 |
|  Fine-tuned VGG16 (from last block)    |  79.82%      |   79.64%  | 23 layers | 14,980,935 |
|  Fine-tuned Inception V3 (from the last 2 inception blocks) |  79.935%   |  79.94% | 315 layers | 22,855,463 |
|  Fine-tuned Inception-ResNet V2 (from the Inception-ResNet-C) | 80.82% | 82.53% | 784 layers | 55,127,271 |
|  Fine-tuned DenseNet 201 (from the last dense block) | **85.8%** | **83.9%**  |  711 layers | 19,309,127 |
|  Fine-tuned Inception V3 (all layers) | 86.92% | 86.826% | _ | _ |
|  Fine-tuned DenseNet 201 (all layers)  | **86.696%** | **87.725%** | _ | _ |
|  Ensemble of fully-fine-tuned Inception V3 and DenseNet 201 | **88.8%** | **88.52%** | _ | _ |


## The Dataset

[The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T,)
