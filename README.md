# SageMaker Transformer Training
This project includes a training script for training a transformer model using sagemaker. The training script supports multi-machine training and checkpointing the best model on a validation in real time. Additionally, this project provides useful functinoality for data processing and loading.

## Features
#### Training:
  * multi-machine training: can use multiple instances of cpu-based machines(ex. ml.c5.2xlarge) to speed up training.
  * saving best checkpoint: evaluates model according to user-provided validation set every user-set number of samples and saves best model
  * printing progress and estimated time for completion of epoch
#### Data:
  * creates dataset from parallel data files
  * loads data in chunks to satisfy memory constraints
  * uses dynamic batching to create batches with roughly equal amounts of tokens
  * allows for data chunks and samples to be sk#ipped to avoid going through same data when starting from a previous checkpoint
## How To Use
The training script is run using SageMaker's PyTorch Estimator. It's necessary to set the *checkpoint_s3_uri* parameter to the s3 bucket you wish to save the checkpoint to. This ensures that the training script uploads checkpoint information in real time during training. When calling the fit method of the Estimator, you provide the s3 bucket that would include the information needed for training such as the training data,validation_data, and sentencepiece model. Feel free to modify the code to suit your needs and if you have any questions feel free to contact me.

#### Contact
email: omaralzamk@gmail.com
