# Medical_Imaging_Projekt

## Setting up a SSH connection
### Linux
- connect to VPN
- open a terminal
- run `ssh user@ctld-n1.cs.technikum-wien.at`

To copy files to the cluster, be in the FHTW VPN, but not connected via ssh and run
`scp <local file path> user@ctld-n1.cs.technikum-wien.at`

### Windows (WinSCP)
- Connect to VPN
- open WinSCP
- create a new connection to user@ctld-n1.cs.technikum-wien.at
- under "Advanced", navigate to "Connection -> Tunnel"
- tick the SSh box on top and fill in your credentials/details
	- add a ssh key, if you have one, otherwise you will be prompted to enter your password

To copy files, use the GUI and drag&drop

## Papers with Code

### On the evaluation of Generative Adversarial Networks

**Topics:**
- Evalutation of GANs
- Commonly computed metrics
- Sample implementation

**Links:**
- Part I:
    - https://towardsdatascience.com/on-the-evaluation-of-generative-adversarial-networks-b056ddcdfd3a#:~:text=The%20most%20common%20metrics%20to,diversity%20(variety%20of%20images).&text=In%20order%20to%20compare%20real,be%20used%20as%20feature%20extractors
- Part II:
    - https://medium.com/mlearning-ai/artificial-intelligence-in-healthcare-part-ii-157301b51c0f

- Repository:
    - https://github.com/aidotse/stylegan2-ada-pytorch


<br>


### ResNet / DenseNet Paper & Website

**Topics:**
- Classification (benign / malignant) with ResNet/DenseNet

**Links:**
- https://github.com/abhinavsagar/Breast-cancer-classification
- https://towardsdatascience.com/convolutional-neural-network-for-breast-cancer-classification-52f1213dcc9


<br>


### Conditional-GAN-with-an-Attention-based-Generator-and-a-3D-Discriminator-for-3D-Medical-Image

**Topics:**
- Classification (benign / malignant) with ResNet/DenseNet

**Links:**
- https://github.com/EuijinMisp/ADESyn
- https://miccai2021.org/openaccess/paperlinks/2021/09/01/100-Paper1250.html








## Research Papers

### CVPR 2021 open access

**Topics:**
- GAN-Based Data Augmentation and Anonymization for Skin-Lesion Analysis: A Critical Review
- Vergleich von verschiedenen GAN Ansätzen

**Link:**
- https://openaccess.thecvf.com/content/CVPR2021W/ISIC/html/Bissoto_GAN-Based_Data_Augmentation_and_Anonymization_for_Skin-Lesion_Analysis_A_Critical_CVPRW_2021_paper.html 


<br>


### DermGAN

**Topics:**
- Synthetic Generation of Clinical Skin Images with Pathology

**Link:**
- https://arxiv.org/abs/1911.08716 


<br>


### Synthetic Data Generation Using Conditional-GAN

**Topics:**
- Modifying the GAN architecture to get more control over the type of data generated
- ist einmal der Towards Data Science Link (erstes Modell) und einmal das verlinkte Github Repo (zweites Modell)

**Link:**
- https://towardsdatascience.com/synthetic-data-generation-using-conditional-gan-45f91542ec6b
- https://gist.github.com/mahmoodm2/519099b45e31f1bce159a6a13e44e1d0

<br>






## Only Repos

### StyleGAN2
- StyleGan2
- https://github.com/NVlabs/stylegan2


<br>


### StainGAN
- StainGAN
- https://github.com/xtarx/StainGAN


<br>


### PGGAN Implementation
- PGGAN
- https://github.com/BradSegal/CXR_PGGAN


<br>


### GANs-for-Medical-Image-Analysis
- verschiedene Repositories von unterschiedlichen GAN Architekturen
- https://github.com/bumuckl/GANs-for-Medical-Image-Analysis


<br>


### Deep Convolution Generative Adversarial Networks
- DCGAN
- https://github.com/pytorch/examples/tree/main/dcgan



<br>


### Improved-Balancing-GAN-Minority-class-Image-Generation
- beinhaltet auch ein Paper (2021)
- https://github.com/GH920/improved-bagan-gp


<br>


### Data Augmentation with Generative Networks for Medical Imaging
- Notebook und älteres Paper (2014)
- https://github.com/apolanco3225/Data-Augmentation-and-Segmentation-with-GANs-for-Medical-Images/blob/master/startGANs.ipynb


<br>


### DCGAN in Tensorflow
- https://github.com/carpedm20/DCGAN-tensorflow


<br>


### Synthetic Medical Images from Dual Generative Adversarial Networks
- Paper included (2017)
- https://github.com/harsha-20/Synthetic-Medical-Images


<br>


### MedImageGAN
- SNGAN which is identical to DCGAN but implements Spectral Normalization to deal with the issue of exploding gradients in the Discriminator
- https://github.com/AlexisMolinaMR/MedImageGAN


<br>


### Generative Adversarial Networks for Image-to-Image Translation on Multi-Contrast MR Images - A Comparison of CycleGAN and UNIT
- https://github.com/simontomaskarlsson/GAN-MRI


<br>









