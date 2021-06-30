# **DFM-Pytorch**

Pytorch implementation of our paper DFM: A Performance Baseline for Deep Feature Matching at [CVPR 2021 Image Matching Workshop](https://image-matching-workshop.github.io/).

[Paper (CVF)](https://openaccess.thecvf.com/content/CVPR2021W/IMW/papers/Efe_DFM_A_Performance_Baseline_for_Deep_Feature_Matching_CVPRW_2021_paper.pdf) | [Paper (arXiv)](https://arxiv.org/abs/2106.07791) <br />
[Presentation (live)](https://youtu.be/9cVV9m_b5Ys?t=9170) | [Presentation (recording)](https://www.youtube.com/watch?v=9oN09WkTwvo)

**Notice** <br />
To reproduce our results given in the paper, use our DFM-Matlab repository. <br /> <br /> *You can get more accurate results (but with fewer features) using Python implementation. It is mainly because MATLAB’s [matchFeatures](https://www.mathworks.com/help/vision/ref/matchfeatures.html) function does not execute ratio test in a bidirectional way, where our Python implementation performs bidirectional ratio test. Nevertheless, we made bidirectionality adjustable in our Python implementation.*

## Setup Environment

**Dependencies** <br />
DFM needs the following dependencies: <br />
(Versions are not strict; however, we have tried DFM with these specific versions.)

- python == 3.7.1
- pytorch == 1.7.1
- torchvision == 0.8.2
- cudatoolkit == 11.0
- matplotlib == 3.3.4
- pillow == 8.2.0
- opencv == 3.4.2
- ipykernel == 5.3.4


We strongly recommend using anaconda. You can simply run the following to create the environment:

````
conda env create -f dfm.yml
conda activte dfm
````

## Enjoy with DFM!
Now you are ready to test DFM by the following command:

````
python DeepFeatureMatcher.py <path_of_imageA> <path_of_imageB>
````

You can make changes to the following arguments:
- Use ***--enable_two_stage*** to enable or disable two stage approach (default: True) <br /> *(Note: Make it enable for planar scenes with significant viewpoint changes, otherwise disable.)*
- Use ***--model*** to change the pre-trained model (default: VGG19) <br /> *(Note: DFM only supports VGG19 and VGG19_BN right now, we plan to add other backbones)*
- Use ***--ratio_th*** to change ratio test thresholds (default: [0.6, 0.6, 0.8, 0.9, 0.95, 1.0]) <br /> *(Note: These ratio test thresholds are for 1st to 5th layer, the last threshold (6th) are for Stage-0 and only usable when --enable_two_stage=True)*
- Use ***--bidirectional*** to enable or disable bidirectional ratio test. (default: True) <br /> *(Note: Make it enable to find more robust matches. Naturally, it should be enabled, make it False is only for similar results with DFM-MATLAB repository since MATLAB's matchFeatures function does not execute ratio test in a bidirectional way)*

## Evaluation
Currently, we have support evaluation only on the HPatches dataset.
You can use our Image Matching Evaluation repository, in which we have support to evaluate SuperPoint, SuperGlue, Patch2Pix, and DFM algorithms on HPatches.
Also, you can use our DFM-Matlab repository to reproduce the results presented in the paper.

