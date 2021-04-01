# DDANet: Dual Decoder Attention Network for Automatic Polyp Segmentation

Authors: [Nikhil Kumar Tomar](https://www.linkedin.com/in/nktomar/), [Debesh Jha](https://scholar.google.com/citations?user=mMTyE68AAAAJ&hl=en), Sharib Ali, Håvard D. Johansen, Dag Johansen, Michael A. Riegler and Pål Halvorsen

## Architecture
The proposed DDANet is fully convolutional network consists of a single encoder and dual decoders. The encoder  consists  of  4  encoder  block  whereas  each  decoder  also  consists  of  4 decoder block. The encoder takes the RGB image as input which passes throughthe shared encoder and then it goes through both the decoders. The first decoder gives the segmentation mask and the  second  decoder gives the original input image in the grayscale format.

![DDANet Architecture](figures/EndoTect.png)

## Quantative Results
| Dataset | DSC |  Mean IOU| Recall | Precision | Mean FPS | Mean Time |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Kvasir Test set | 0.8576 | 0.7800 | 0.8880 | 0.8643 | 70.23445 | 0.014238
| Organiser's Test set | 0.7010 | 0.7874 | 0.7987 | 0.8577 | 69.59296 | 0.014369

## Qualitative Results
![Qualitative Results](figures/figure_name.png)

## Citation
Please cite our paper if you find the work useful: 
<pre>
@inproceedings{tomar2020ddanet,
  title={DDANet: Dual Decoder Attention Network for Automatic Polyp Segmentation},
  author={Tomar, Nikhil Kumar and Jha, Debesh and Ali, Sharib and Johansen, H{\aa}vard D and Johansen, Dag and Riegler, Michael A and Halvorsen, P{\aa}l},
  booktitle={ICPR International Workshop and Challenges},
  year={2021}
}
</pre>

## Contact
please contact nikhilroxtomar@gmail.com and debesh@simula.no for any further questions. 
