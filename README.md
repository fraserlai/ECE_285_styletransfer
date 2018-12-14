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

## Training the evaluation code:


### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

