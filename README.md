# Generalized-Adversarially-Learned-Inference
This repository contains the code for the paper: [Generalized Adversarially Learned Inference](https://arxiv.org/abs/2006.08089), published in AAAI, 2021.

## Abstract
Allowing effective inference of latent vectors while training GANs can greatly increase their applicability in various downstream tasks. Recent approaches, such as ALI and BiGAN frameworks, develop methods of inference of latent variables in GANs by adversarially training an image generator along with an encoder to match two joint distributions of image and latent vector pairs. We generalize these approaches to incorporate multiple layers of feedback on reconstructions, self-supervision, and other forms of supervision based on prior or learned knowledge about the desired solutions. We achieve this by modifying the discriminator's objective to correctly identify more than two joint distributions of tuples of an arbitrary number of random variables consisting of images, latent vectors, and other variables generated through auxiliary tasks, such as reconstruction and inpainting or as outputs of suitable pre-trained models. We design a non-saturating maximization objective for the generator-encoder pair and prove that the resulting adversarial game corresponds to a global optimum that simultaneously matches all the distributions. Within our proposed framework, we introduce a novel set of techniques for providing self-supervised feedback to the model based on properties, such as patch-level correspondence and cycle consistency of reconstructions. Through comprehensive experiments, we demonstrate the efficacy, scalability, and flexibility of the proposed approach for a variety of tasks. 


## References
If you use the code, please cite the following paper:

```
@article{Dandi_Bharadhwaj_Kumar_Rai_2021, title={Generalized Adversarially Learned Inference}, volume={35}, url={https://ojs.aaai.org/index.php/AAAI/article/view/16883}, number={8}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Dandi, Yatin and Bharadhwaj, Homanga and Kumar, Abhishek and Rai, Piyush}, year={2021}, month={May}, pages={7185-7192} }
```
