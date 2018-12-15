# ECE_285_styletransfer

The project focus on how to keep the original text on comic images and perform style transfer on comic images. 

## Getting Started
The following instrustion is to deploy the system on [UCSD ITS](http://go.ucsd.edu/2CZladZ.) computing cluster.
The docker image is: ```fraserlai/285_project:v1``` or ```ucsdets/instructional:ets-pytorch-py3-latest```

```
prep ee285f
```

```
launch-pytorch-gpu.sh -i fraserlai/285_project:v1
```
or 

```
launch-py3torch-gpu-cuda9.sh
```

### Installing all requirements
Run the following commend in terminal
```
pip install -r requirements.txt --user
```

## Running the evaluation code:

User the juypyter notebook: ```eval_notebook.ipynb``` in ```src```.

The following are parameters for style image, content iamge, model weight and final output.
```
style_img_path = "./data/9styles/composition_vii.jpg"
content_img_path = "./data/content/Ben Reilly - Scarlet Spider (2017-) 016-010.jpg"
style_transfer_output_path = "./output/out.png"

test_data_path = './data/content'
checkpoint_path = './checkpoints_total/model_220.pth'
output_dir_box = './textresult/box'
output_dir_txt = './textresult/txt'
output_dir_pic = './textresult/pic'
 
mask_path = './textresult/txt/Ben Reilly - Scarlet Spider (2017-) 016-010.txt'
final_output_path = "./output/final.png"
```

## Training the weight:
### Training the style transfer network code:
Train your own VGG net by using COCO dataset: ```Train_StyleTransfer_v2.ipynb```.
The dataset should put under ```./dataset/train2017```

### Training the EAST text detection code:
Use the juypyter notebook: ```text_localization_train.ipynb``` in ```kuanweichen_workspace/285_project```

The trained network parameters will be saved every 20 epochs under ```kuanweichen_workspace/checkpoints_total```, with name ```model_XX``` where ```XX``` being the epoch number.



