# Getting the Numbers—Modelling Multi-class Object Counting in Dense and Varied Scenes

Density map estimation enables accurate object counting in heavily occluded, and densely packed scenes where detection-based counting fails. In multi-class density estimation, class awareness can be introduced by modelling classes non-exclusively, better reflecting crowded and visually ambiguous contexts.

This repository contains the code used in our work.

> [!NOTE]
> This work was supported by the UKRI AI Centre for Doctoral Training in Sustainable Understandable agri-food Systems Transformed by Artificial INtelligence (SUSTAIN) [grant reference: EP/Y03063X/1]. 
> 
> Published as part of the International Joint Conference on Computational Intelligence at IEEE World Congress of Computational Intelligence 2026.

The paper is available on arXiv, and on the [University of Lincoln Figshare Site](https://repository.lincoln.ac.uk/articles/conference_contribution/Getting_the_Numbers_Right_Modelling_Multi-Class_Object_Counting_in_Dense_and_Varied_Scenes/32025327).

## Datasets

- [Hicks Biodiversity Dataset](https://doi.org/10.5061/dryad.63xsj3v34). Gathered for biodiversity object detection, by Damien Hicks et al. in the *Ecological Solutions and Evidence* 2021 article *[Deep learning object detection to estimate the nectar sugar mass of flowering vegetation](https://doi.org/10.1002/2688-8319.12099)*.

- [VisDrone](https://github.com/VisDrone/VisDrone-Dataset), a drone based detection challenge for crowded urban areas. We use VisDrone-DET set, so to re-create our results, the trainset, valset and testset-dev must be downloaded. Their GitHub provides links for both Google Drive and BaiduYun downloads.

- [iSAID](https://captain-whu.github.io/iSAID/), a remote sensing segmentation dataset using the images of [DOTA-V1.0](https://captain-whu.github.io/DOTA/index.html).

You must download the datasets, and investigate the scripts within [`convert_datasets`](./convert_datasets/). The scripts are seeded, so will produce the same datasets used in this paper. Due to no suitable test/challenge set for iSAID and Hicks et al., those datasets are shuffled to a custom split. Ensure no seeds are changed.
