"""Reference database (IEEE-like, numbered by first citation).

Entries marked  # V  were checked against an online record while the thesis was
prepared (title, authors, venue/arXiv id).  The remaining entries are standard
literature cited from the author's knowledge; their bibliographic details
(page ranges in particular) should be re-checked against the publisher record
before submission.  *...* marks italics.
"""

REFS = {
    "prelim": 'M. S. Yadkar and S. Kurnaz, "3D tumor segmentation and classification in multimodal imaging using 3D U-Net and GANs," manuscript submitted to *Applied Sciences* (MDPI), 2025.',
    # --- segmentation architectures ------------------------------------
    "fcn": 'J. Long, E. Shelhamer, and T. Darrell, "Fully convolutional networks for semantic segmentation," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2015, pp. 3431–3440.',
    "unet": 'O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation," in *Medical Image Computing and Computer-Assisted Intervention (MICCAI)*, ser. LNCS, vol. 9351, 2015, pp. 234–241.',
    "unet3d": 'Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, and O. Ronneberger, "3D U-Net: Learning dense volumetric segmentation from sparse annotation," in *MICCAI*, ser. LNCS, vol. 9901, 2016, pp. 424–432.',
    "vnet": 'F. Milletari, N. Navab, and S.-A. Ahmadi, "V-Net: Fully convolutional neural networks for volumetric medical image segmentation," in *Proc. Int. Conf. 3D Vision (3DV)*, 2016, pp. 565–571.',
    "unetpp": 'Z. Zhou, M. M. R. Siddiquee, N. Tajbakhsh, and J. Liang, "UNet++: A nested U-Net architecture for medical image segmentation," in *Deep Learning in Medical Image Analysis (DLMIA)*, ser. LNCS, vol. 11045, 2018, pp. 3–11.',
    "attunet": 'O. Oktay *et al.*, "Attention U-Net: Learning where to look for the pancreas," arXiv:1804.03999, 2018.',
    "attgated": 'J. Schlemper *et al.*, "Attention gated networks: Learning to leverage salient regions in medical images," *Med. Image Anal.*, vol. 53, pp. 197–207, 2019.',
    "nnunet": 'F. Isensee, P. F. Jäger, S. A. A. Kohl, J. Petersen, and K. H. Maier-Hein, "nnU-Net: A self-configuring method for deep learning-based biomedical image segmentation," *Nature Methods*, vol. 18, no. 2, pp. 203–211, 2021.',
    "nonew": 'F. Isensee, P. Kickingereder, W. Wick, M. Bendszus, and K. H. Maier-Hein, "No New-Net," arXiv:1809.10483, 2018.',  # V
    "nnunet_brats20": 'F. Isensee, P. F. Jäger, P. M. Full, P. Vollmuth, and K. H. Maier-Hein, "nnU-Net for brain tumor segmentation," arXiv:2011.00848, 2020.',  # V
    "myronenko": 'A. Myronenko, "3D MRI brain tumor segmentation using autoencoder regularization," arXiv:1810.11654, 2018 (BrainLes 2018).',  # V
    "deepmedic": 'K. Kamnitsas *et al.*, "Efficient multi-scale 3D CNN with fully connected CRF for accurate brain lesion segmentation," *Med. Image Anal.*, vol. 36, pp. 61–78, 2017.',
    "pereira": 'S. Pereira, A. Pinto, V. Alves, and C. A. Silva, "Brain tumor segmentation using convolutional neural networks in MRI images," *IEEE Trans. Med. Imaging*, vol. 35, no. 5, pp. 1240–1251, 2016.',
    "havaei": 'M. Havaei *et al.*, "Brain tumor segmentation with deep neural networks," *Med. Image Anal.*, vol. 35, pp. 18–31, 2017.',
    "unetr": 'A. Hatamizadeh *et al.*, "UNETR: Transformers for 3D medical image segmentation," in *Proc. IEEE/CVF Winter Conf. Appl. Comput. Vis. (WACV)*, 2022, pp. 574–584.',
    "swinunetr": 'A. Hatamizadeh, V. Nath, Y. Tang, D. Yang, H. R. Roth, and D. Xu, "Swin UNETR: Swin transformers for semantic segmentation of brain tumors in MRI images," arXiv:2201.01266, 2022.',  # V
    "transbts": 'W. Wang, C. Chen, M. Ding, J. Li, H. Yu, and S. Zha, "TransBTS: Multimodal brain tumor segmentation using transformer," arXiv:2103.04430, 2021.',  # V
    "transunet": 'J. Chen *et al.*, "TransUNet: Transformers make strong encoders for medical image segmentation," arXiv:2102.04306, 2021.',
    "vit": 'A. Dosovitskiy *et al.*, "An image is worth 16x16 words: Transformers for image recognition at scale," in *Int. Conf. Learn. Represent. (ICLR)*, 2021.',
    "transformer": 'A. Vaswani *et al.*, "Attention is all you need," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017.',
    "monai": 'M. J. Cardoso *et al.*, "MONAI: An open-source framework for deep learning in healthcare," arXiv:2211.02701, 2022.',
    # --- BraTS / data --------------------------------------------------
    "brats15": 'B. H. Menze *et al.*, "The multimodal brain tumor image segmentation benchmark (BRATS)," *IEEE Trans. Med. Imaging*, vol. 34, no. 10, pp. 1993–2024, Oct. 2015.',  # V
    "brats17": 'S. Bakas *et al.*, "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features," *Scientific Data*, vol. 4, art. 170117, 2017.',  # V
    "brats18": 'S. Bakas *et al.*, "Identifying the best machine learning algorithms for brain tumor segmentation, progression assessment, and overall survival prediction in the BRATS challenge," arXiv:1811.02629, 2018.',  # V
    "brats21": 'U. Baid *et al.*, "The RSNA-ASNR-MICCAI BraTS 2021 benchmark on brain tumor segmentation and radiogenomic classification," arXiv:2107.02314, 2021.',  # V
    "tcia": 'K. Clark *et al.*, "The Cancer Imaging Archive (TCIA): Maintaining and operating a public information repository," *J. Digit. Imaging*, vol. 26, no. 6, pp. 1045–1057, 2013.',
    "msd": 'M. Antonelli *et al.*, "The Medical Segmentation Decathlon," *Nature Commun.*, vol. 13, art. 4128, 2022.',
    "n4": 'N. J. Tustison *et al.*, "N4ITK: Improved N3 bias correction," *IEEE Trans. Med. Imaging*, vol. 29, no. 6, pp. 1310–1320, 2010.',
    "nyul": 'L. G. Nyúl and J. K. Udupa, "On standardizing the MR image intensity scale," *Magn. Reson. Med.*, vol. 42, no. 6, pp. 1072–1081, 1999.',
    "oakden": 'L. Oakden-Rayner, "Exploring large-scale public medical image datasets," *Acad. Radiol.*, vol. 27, no. 1, pp. 106–112, 2020.',
    # --- clinical ------------------------------------------------------
    "who2016": 'D. N. Louis *et al.*, "The 2016 World Health Organization classification of tumors of the central nervous system: A summary," *Acta Neuropathol.*, vol. 131, no. 6, pp. 803–820, 2016.',
    "who2021": 'D. N. Louis *et al.*, "The 2021 WHO classification of tumors of the central nervous system: A summary," *Neuro-Oncology*, vol. 23, no. 8, pp. 1231–1251, 2021.',
    "stupp": 'R. Stupp *et al.*, "Radiotherapy plus concomitant and adjuvant temozolomide for glioblastoma," *N. Engl. J. Med.*, vol. 352, no. 10, pp. 987–996, 2005.',
    "rano": 'P. Y. Wen *et al.*, "Updated response assessment criteria for high-grade gliomas: Response Assessment in Neuro-Oncology working group," *J. Clin. Oncol.*, vol. 28, no. 11, pp. 1963–1972, 2010.',
    # --- surveys -------------------------------------------------------
    "litjens": 'G. Litjens *et al.*, "A survey on deep learning in medical image analysis," *Med. Image Anal.*, vol. 42, pp. 60–88, 2017.',
    "shen": 'D. Shen, G. Wu, and H.-I. Suk, "Deep learning in medical image analysis," *Annu. Rev. Biomed. Eng.*, vol. 19, pp. 221–248, 2017.',
    "lecun": 'Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436–444, 2015.',
    "ghaffari": 'M. Ghaffari, A. Sowmya, and R. Oliver, "Automated brain tumor segmentation using multimodal brain scans: A survey based on models submitted to the BraTS 2012–2018 challenges," *IEEE Rev. Biomed. Eng.*, vol. 13, pp. 156–168, 2020.',
    "bernal": 'J. Bernal *et al.*, "Deep convolutional neural networks for brain image analysis on magnetic resonance imaging: A review," *Artif. Intell. Med.*, vol. 95, pp. 64–81, 2019.',
    "yi_gan": 'X. Yi, E. Walia, and P. Babyn, "Generative adversarial network in medical imaging: A review," *Med. Image Anal.*, vol. 58, art. 101552, 2019.',
    "zhou_fusion": 'T. Zhou, S. Ruan, and S. Canu, "A review: Deep learning for medical image segmentation using multi-modality fusion," *Array*, vol. 3–4, art. 100004, 2019.',
    # --- GANs ----------------------------------------------------------
    "gan": 'I. Goodfellow *et al.*, "Generative adversarial nets," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 27, 2014.',
    "dcgan": 'A. Radford, L. Metz, and S. Chintala, "Unsupervised representation learning with deep convolutional generative adversarial networks," arXiv:1511.06434, 2015.',
    "pix2pix": 'P. Isola, J.-Y. Zhu, T. Zhou, and A. A. Efros, "Image-to-image translation with conditional adversarial networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2017, pp. 1125–1134.',
    "luc": 'P. Luc, C. Couprie, S. Chintala, and J. Verbeek, "Semantic segmentation using adversarial networks," arXiv:1611.08408, 2016.',  # V
    "segan": 'Y. Xue, T. Xu, H. Zhang, L. R. Long, and X. Huang, "SegAN: Adversarial network with multi-scale L1 loss for medical image segmentation," *Neuroinformatics*, vol. 16, no. 3–4, pp. 383–392, 2018.',  # V
    "lsgan": 'X. Mao, Q. Li, H. Xie, R. Y. K. Lau, Z. Wang, and S. P. Smolley, "Least squares generative adversarial networks," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2794–2802.',
    "wgan": 'M. Arjovsky, S. Chintala, and L. Bottou, "Wasserstein generative adversarial networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 70, 2017, pp. 214–223.',
    "wgangp": 'I. Gulrajani, F. Ahmed, M. Arjovsky, V. Dumoulin, and A. Courville, "Improved training of Wasserstein GANs," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017.',
    "sn": 'T. Miyato, T. Kataoka, M. Koyama, and Y. Yoshida, "Spectral normalization for generative adversarial networks," in *Int. Conf. Learn. Represent. (ICLR)*, 2018.',
    "arjovsky_bottou": 'M. Arjovsky and L. Bottou, "Towards principled methods for training generative adversarial networks," in *Int. Conf. Learn. Represent. (ICLR)*, 2017.',
    "sonderby": 'C. K. Sønderby, J. Caballero, L. Theis, W. Shi, and F. Huszár, "Amortised MAP inference for image super-resolution," arXiv:1610.04490, 2016.',
    "salimans": 'T. Salimans *et al.*, "Improved techniques for training GANs," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 29, 2016.',
    "mescheder": 'L. Mescheder, A. Geiger, and S. Nowozin, "Which training methods for GANs do actually converge?" in *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 80, 2018, pp. 3481–3490.',
    "ttur": 'M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 30, 2017.',
    "odena": 'A. Odena, V. Dumoulin, and C. Olah, "Deconvolution and checkerboard artifacts," *Distill*, 2016.',
    "gan_seg_liver": 'D. Yang *et al.*, "Automatic liver segmentation using an adversarial image-to-image network," in *MICCAI*, ser. LNCS, vol. 10435, 2017, pp. 507–515.',
    "moeskops": 'P. Moeskops, M. Veta, M. W. Lafarge, K. A. J. Eppenhof, and J. P. W. Pluim, "Adversarial training and dilated convolutions for brain MRI segmentation," in *DLMIA*, ser. LNCS, vol. 10553, 2017, pp. 56–64.',
    "rezaei": 'M. Rezaei, K. Harmuth, W. Gierke, T. Kellermeier, M. Fischer, H. Yang, and C. Meinel, "A conditional adversarial network for semantic segmentation of brain tumor," arXiv:1708.05227, 2017.',
    "kohl": 'S. Kohl *et al.*, "Adversarial networks for the detection of aggressive prostate cancer," arXiv:1702.08014, 2017.',
    "dong_ganseg": 'X. Dong *et al.*, "Automatic multiorgan segmentation in thorax CT images using U-net-GAN," *Med. Phys.*, vol. 46, no. 5, pp. 2157–2168, 2019.',
    "shin_synth": 'H.-C. Shin *et al.*, "Medical image synthesis for data augmentation and anonymization using generative adversarial networks," in *Simulation and Synthesis in Medical Imaging (SASHIMI)*, ser. LNCS, vol. 11037, 2018, pp. 1–11.',
    "fridadar": 'M. Frid-Adar, I. Diamant, E. Klang, M. Amitai, J. Goldberger, and H. Greenspan, "GAN-based synthetic medical image augmentation for increased CNN performance in liver lesion classification," *Neurocomputing*, vol. 321, pp. 321–331, 2018.',
    "bowles": 'C. Bowles *et al.*, "GAN augmentation: Augmenting training data using generative adversarial networks," arXiv:1810.10863, 2018.',
    # --- losses --------------------------------------------------------
    "dice1945": 'L. R. Dice, "Measures of the amount of ecologic association between species," *Ecology*, vol. 26, no. 3, pp. 297–302, 1945.',
    "tversky1977": 'A. Tversky, "Features of similarity," *Psychol. Rev.*, vol. 84, no. 4, pp. 327–352, 1977.',
    "tversky": 'S. S. M. Salehi, D. Erdogmus, and A. Gholipour, "Tversky loss function for image segmentation using 3D fully convolutional deep networks," in *Machine Learning in Medical Imaging (MLMI)*, ser. LNCS, vol. 10541, 2017 (arXiv:1706.05721).',  # V
    "gdl": 'C. H. Sudre, W. Li, T. Vercauteren, S. Ourselin, and M. J. Cardoso, "Generalised Dice overlap as a deep learning loss function for highly unbalanced segmentations," in *DLMIA*, ser. LNCS, vol. 10553, 2017, pp. 240–248.',  # V
    "focaltv": 'N. Abraham and N. M. Khan, "A novel focal Tversky loss function with improved Attention U-Net for lesion segmentation," arXiv:1810.07842, 2018 (ISBI 2019).',  # V
    "focal": 'T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2017, pp. 2980–2988.',
    "combo": 'S. A. Taghanaki *et al.*, "Combo loss: Handling input and output imbalance in multi-organ segmentation," *Comput. Med. Imaging Graph.*, vol. 75, pp. 24–33, 2019.',
    "odyssey": 'J. Ma *et al.*, "Loss odyssey in medical image segmentation," *Med. Image Anal.*, vol. 71, art. 102035, 2021.',
    "eelbode": 'T. Eelbode *et al.*, "Optimization for medical image segmentation: Theory and practice when evaluating with Dice score or Jaccard index," *IEEE Trans. Med. Imaging*, vol. 39, no. 11, pp. 3679–3690, 2020.',
    "bertels": 'J. Bertels *et al.*, "Optimizing the Dice score and Jaccard index for medical image segmentation: Theory and practice," in *MICCAI*, ser. LNCS, vol. 11765, 2019, pp. 92–100.',
    "lovasz": 'M. Berman, A. R. Triki, and M. B. Blaschko, "The Lovász-softmax loss: A tractable surrogate for the optimization of the intersection-over-union measure in neural networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 4413–4421.',
    "boundary": 'H. Kervadec *et al.*, "Boundary loss for highly unbalanced segmentation," in *Proc. Int. Conf. Med. Imaging Deep Learn. (MIDL)*, PMLR vol. 102, 2019, pp. 285–296.',
    "deepsup": 'C.-Y. Lee, S. Xie, P. Gallagher, Z. Zhang, and Z. Tu, "Deeply-supervised nets," in *Proc. Int. Conf. Artif. Intell. Statist. (AISTATS)*, PMLR vol. 38, 2015, pp. 562–570.',
    # --- building blocks / optimisation --------------------------------
    "resnet": 'K. He, X. Zhang, S. Ren, and J. Sun, "Deep residual learning for image recognition," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016, pp. 770–778.',
    "heinit": 'K. He, X. Zhang, S. Ren, and J. Sun, "Delving deep into rectifiers: Surpassing human-level performance on ImageNet classification," in *Proc. IEEE Int. Conf. Comput. Vis. (ICCV)*, 2015, pp. 1026–1034.',
    "bn": 'S. Ioffe and C. Szegedy, "Batch normalization: Accelerating deep network training by reducing internal covariate shift," in *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 37, 2015, pp. 448–456.',
    "instnorm": 'D. Ulyanov, A. Vedaldi, and V. Lempitsky, "Instance normalization: The missing ingredient for fast stylization," arXiv:1607.08022, 2016.',
    "lrelu": 'A. L. Maas, A. Y. Hannun, and A. Y. Ng, "Rectifier nonlinearities improve neural network acoustic models," in *ICML Workshop on Deep Learning for Audio, Speech and Language Processing*, 2013.',
    "dropout": 'N. Srivastava, G. Hinton, A. Krizhevsky, I. Sutskever, and R. Salakhutdinov, "Dropout: A simple way to prevent neural networks from overfitting," *J. Mach. Learn. Res.*, vol. 15, pp. 1929–1958, 2014.',
    "se": 'J. Hu, L. Shen, and G. Sun, "Squeeze-and-excitation networks," in *Proc. IEEE Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2018, pp. 7132–7141.',
    "cbam": 'S. Woo, J. Park, J.-Y. Lee, and I. S. Kweon, "CBAM: Convolutional block attention module," in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, ser. LNCS, vol. 11211, 2018, pp. 3–19.',
    "erf": 'W. Luo, Y. Li, R. Urtasun, and R. Zemel, "Understanding the effective receptive field in deep convolutional neural networks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 29, 2016.',
    "adam": 'D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Int. Conf. Learn. Represent. (ICLR)*, 2015.',
    "adamw": 'I. Loshchilov and F. Hutter, "Decoupled weight decay regularization," in *Int. Conf. Learn. Represent. (ICLR)*, 2019.',
    "deeplab": 'L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille, "DeepLab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected CRFs," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 40, no. 4, pp. 834–848, 2018.',
    "gradclip": 'R. Pascanu, T. Mikolov, and Y. Bengio, "On the difficulty of training recurrent neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 28, 2013, pp. 1310–1318.',
    "amp": 'P. Micikevicius *et al.*, "Mixed precision training," in *Int. Conf. Learn. Represent. (ICLR)*, 2018.',
    "pytorch": 'A. Paszke *et al.*, "PyTorch: An imperative style, high-performance deep learning library," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 32, 2019, pp. 8024–8035.',
    "sklearn": 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *J. Mach. Learn. Res.*, vol. 12, pp. 2825–2830, 2011.',
    "gbm": 'J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Ann. Statist.*, vol. 29, no. 5, pp. 1189–1232, 2001.',
    "shorten": 'C. Shorten and T. M. Khoshgoftaar, "A survey on image data augmentation for deep learning," *J. Big Data*, vol. 6, art. 60, 2019.',
    "hemis": 'M. Havaei, N. Guizard, N. Chapados, and Y. Bengio, "HeMIS: Hetero-modal image segmentation," in *MICCAI*, ser. LNCS, vol. 9901, 2016, pp. 469–477.',
    "hyperdense": 'J. Dolz, K. Gopinath, J. Yuan, H. Lombaert, C. Desrosiers, and I. Ben Ayed, "HyperDense-Net: A hyper-densely connected CNN for multi-modal image segmentation," *IEEE Trans. Med. Imaging*, vol. 38, no. 5, pp. 1116–1126, 2019.',
    "rfnet": 'Y. Ding, X. Yu, and Y. Yang, "RFNet: Region-aware fusion network for incomplete multi-modal brain tumor segmentation," in *Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV)*, 2021, pp. 3975–3984.',
    # --- metrics / statistics -----------------------------------------
    "taha": 'A. A. Taha and A. Hanbury, "Metrics for evaluating 3D medical image segmentation: Analysis, selection, and tool," *BMC Med. Imaging*, vol. 15, art. 29, 2015.',  # V
    "metricsreloaded": 'L. Maier-Hein *et al.*, "Metrics reloaded: Recommendations for image analysis validation," *Nature Methods*, vol. 21, no. 2, pp. 195–212, 2024.',  # V
    "reinke": 'A. Reinke *et al.*, "Understanding metric-related pitfalls in image analysis validation," *Nature Methods*, vol. 21, no. 2, pp. 182–194, 2024.',
    "hausdorff": 'D. P. Huttenlocher, G. A. Klanderman, and W. J. Rucklidge, "Comparing images using the Hausdorff distance," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 15, no. 9, pp. 850–863, 1993.',
    "mh_rankings": 'L. Maier-Hein *et al.*, "Why rankings of biomedical image analysis competitions should be interpreted with care," *Nature Commun.*, vol. 9, art. 5217, 2018.',
    "efron79": 'B. Efron, "Bootstrap methods: Another look at the jackknife," *Ann. Statist.*, vol. 7, no. 1, pp. 1–26, 1979.',
    "efron_tib": 'B. Efron and R. J. Tibshirani, *An Introduction to the Bootstrap*. New York, NY, USA: Chapman & Hall, 1993.',
    "diciccio": 'T. J. DiCiccio and B. Efron, "Bootstrap confidence intervals," *Statist. Sci.*, vol. 11, no. 3, pp. 189–228, 1996.',
    "wilcoxon": 'F. Wilcoxon, "Individual comparisons by ranking methods," *Biometrics Bull.*, vol. 1, no. 6, pp. 80–83, 1945.',
    "holm": 'S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.',
    "cliff": 'N. Cliff, "Dominance statistics: Ordinal analyses to answer ordinal questions," *Psychol. Bull.*, vol. 114, no. 3, pp. 494–509, 1993.',
    "cohen": 'J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Hillsdale, NJ, USA: Lawrence Erlbaum, 1988.',
    "demsar": 'J. Demšar, "Statistical comparisons of classifiers over multiple data sets," *J. Mach. Learn. Res.*, vol. 7, pp. 1–30, 2006.',
    "benavoli": 'A. Benavoli, G. Corani, J. Demšar, and M. Zaffalon, "Time for a change: A tutorial for comparing multiple classifiers through Bayesian analysis," *J. Mach. Learn. Res.*, vol. 18, no. 77, pp. 1–36, 2017.',
    "dietterich": 'T. G. Dietterich, "Approximate statistical tests for comparing supervised classification learning algorithms," *Neural Comput.*, vol. 10, no. 7, pp. 1895–1923, 1998.',
    "kish": 'L. Kish, *Survey Sampling*. New York, NY, USA: Wiley, 1965.',
    "button": 'K. S. Button *et al.*, "Power failure: Why small sample size undermines the reliability of neuroscience," *Nature Rev. Neurosci.*, vol. 14, no. 5, pp. 365–376, 2013.',
    "varoquaux": 'G. Varoquaux, "Cross-validation failure: Small sample sizes lead to large error bars," *NeuroImage*, vol. 180, pp. 68–77, 2018.',
    "cawley": 'G. C. Cawley and N. L. C. Talbot, "On over-fitting in model selection and subsequent selection bias in performance evaluation," *J. Mach. Learn. Res.*, vol. 11, pp. 2079–2107, 2010.',
    "kaufman": 'S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Trans. Knowl. Discov. Data*, vol. 6, no. 4, art. 15, 2012.',
    "asa": 'R. L. Wasserstein and N. A. Lazar, "The ASA statement on p-values: Context, process, and purpose," *Amer. Statist.*, vol. 70, no. 2, pp. 129–133, 2016.',
    "guo": 'C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, "On calibration of modern neural networks," in *Proc. Int. Conf. Mach. Learn. (ICML)*, PMLR vol. 70, 2017, pp. 1321–1330.',
    "naeini": 'M. P. Naeini, G. Cooper, and M. Hauskrecht, "Obtaining well calibrated probabilities using Bayesian binning," in *Proc. AAAI Conf. Artif. Intell.*, 2015, pp. 2901–2907.',
    "hanley": 'J. A. Hanley and B. J. McNeil, "The meaning and use of the area under a receiver operating characteristic (ROC) curve," *Radiology*, vol. 143, no. 1, pp. 29–36, 1982.',
    "delong": 'E. R. DeLong, D. M. DeLong, and D. L. Clarke-Pearson, "Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach," *Biometrics*, vol. 44, no. 3, pp. 837–845, 1988.',
    # --- radiomics / grading ------------------------------------------
    "gillies": 'R. J. Gillies, P. E. Kinahan, and H. Hricak, "Radiomics: Images are more than pictures, they are data," *Radiology*, vol. 278, no. 2, pp. 563–577, 2016.',
    "aerts": 'H. J. W. L. Aerts *et al.*, "Decoding tumour phenotype by noninvasive imaging using a quantitative radiomics approach," *Nature Commun.*, vol. 5, art. 4006, 2014.',
    "pyradiomics": 'J. J. M. van Griethuysen *et al.*, "Computational radiomics system to decode the radiographic phenotype," *Cancer Res.*, vol. 77, no. 21, pp. e104–e107, 2017.',
    "ibsi": 'A. Zwanenburg *et al.*, "The Image Biomarker Standardization Initiative: Standardized quantitative radiomics for high-throughput image-based phenotyping," *Radiology*, vol. 295, no. 2, pp. 328–338, 2020.',
    "cho": 'H.-H. Cho, S.-H. Lee, J. Kim, and H. Park, "Classification of the glioma grading using radiomics analysis," *PeerJ*, vol. 6, art. e5982, 2018.',
    "decuyper": 'M. Decuyper, S. Bonte, K. Deblaere, and R. Van Holen, "Automated MRI based pipeline for segmentation and prediction of grade, IDH mutation and 1p19q co-deletion in glioma," *Comput. Med. Imaging Graph.*, vol. 88, art. 101831, 2021.',
}
