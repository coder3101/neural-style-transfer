# Neural Style Transfer

Neural Style Transfer is a simple method of transferring the style of one image to other. To Know more about this and how it works. Give [this paper](https://arxiv.org/pdf/1508.06576.pdf) a read.

**WARNING : YOU NEED A GPU TO RUN THIS LOCALLY**

**If you don't have a GPU we have a Colab Notebook [here](https://colab.research.google.com/drive/13cIYAyAazgHFTo0nCyFt3vsF4nRSqgr7)**

> Make Sure to choose GPU Runtime : 
>
> Runtime > Change Runtime > Hardware Accelerator > GPU

---

### Getting Dependencies

Clone the repository and in that directory follow the steps below.

To get all the dependencies run the following :

```bash
pip install -r requirements.txt
```

Actually there are only 3 dependencies so if you have them you can safely ignore the above step. Dependencies are : `tensorflow` , `matplotlib` and `numpy` .



---

### Getting ready with Pictures

You need only two pictures to run the model. 

1. Content Image : Generated Picture will resemble this image
2. Style Image : Generated Picure will capture style from this Image

You can choose any Image you wish. It doesn't have any restrictions on size or format.  Any format that decomposes to RGB will work like jpg, jpeg etcetra.

> LARGER THE CONTENT IMAGE, LONGER IT WILL TAKE FOR THE MODEL TO CONVERGE.

Precisely For Content Image of 200x400 Pixels. Model Needs to Optimize 200x400x3 Values

As an Example Let's take this Picture as Content Image.

<p align="center">

<img src="https://github.com/coder3101/neural-style-transfer/raw/master/content.jpg"/>

</p>

Style Image can also be of any size and any RGB decomposing format. We crop or pad the style image to match the dimension of Content Image. However it is recommended to use Style Image of the almost same dimension as content Image.

As an Example Let's take this Picture as Style Image.

<p align="center">

<img src="https://github.com/coder3101/neural-style-transfer/raw/master/style.jpg"/>

</p> 

---

### Getting ready for Training

I have carely tuned all the hyper parameters for you.

In order to get started with the training phase. You need to run a script called `generate.py` . In the end the script will generate an image named `styled_image.jpg`. This is the styled image of the model.

Simply run :

```bash
python generate.py --content_image path/to/content/image.jpg \ 
			   	   --style_image path/to/style/image.jpg
```

To get fine control over the results :

```bash
python generate.py --content_image path/to/content/image.jpg \
				--style_image path/to/style/image.jpg \
				--gen_file_name name_of_output_image.jpg \
				--epochs 4512 \
				--random_init \
				--learning_rate 0.094 \
				--content_factor 0.0001 \
				--style_factor 1 \
				--optimizer 'rms' \
				--loss_after 500
```

To get to know what each argument does please run :

```bash
python generate.py --help
```

#### Important Arguments

**--random_init** is a flag when set causes the image to generate from noise. By default to reduce training time we set this flag to false. This makes applies the style directly to content image

**--content_factor** determines how close should generated image be to content image. Geneally keep it lower than 1. You don't want the generated image to look exactly the same as content image.

**--style_factor** determines how much style to apply to original image. Keep this value over 10. Values less than these do no generally show the style dominance in content image.

> TO GET MORE CONTROL OVER THE LAYER TO CHOOSE AND STYLE WEIGHTS YOU SHOULD DIRECTLY MODIFY THE CODE IN `generate.py`

After trainingon above content and style image with default hyper parameters. We got :

<p align="center">

<img src="https://github.com/coder3101/neural-style-transfer/raw/master/styled_image.jpg"/>

</p>

Hmm. Looks like an Oil Painting..

***

## Happy Coding...

From Ashar (coder3101)

