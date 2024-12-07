# Introduction 
Machine Learning Projects in DotNet

# Folder FASSTEXT

dotnet add package Panlingo.LanguageIdentification.CLD3 --version 0.0.0.18

https://github.com/gluschenko/language-identification/tree/master

### Install
```
sudo apt -y install protobuf-compiler libprotobuf-dev nuget
 
wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.de
sudo apt-get update &&   sudo apt-get install -y dotnet-sdk-8.0
 ```
# Models
```
curl --location -o /models/fasttext176.bin https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
curl --location -o /models/fasttext217.bin https://huggingface.co/facebook/fasttext-language-identification/resolve/main/model.bin?download=true
``` 
### build app in LInux
```
dotnet new create console -n lang_detector
dotnet add package Panlingo.LanguageIdentification.FastText
dotnet add package  Mosaik.Core --version 24.8.51117
dotnet run
dotnet lang_detector.dll
``` 
### https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
 

# Folder CLD_3

dotnet add package Panlingo.LanguageIdentification.CLD3 --version 0.0.0.18
https://github.com/gluschenko/language-identification/tree/master

### Install
```
sudo apt -y install protobuf-compiler libprotobuf-dev nuget
 
wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
rm packages-microsoft-prod.de
sudo apt-get update &&   sudo apt-get install -y dotnet-sdk-8.0
```
 
 ### Build app in LInux
 ```
 dotnet new create console -n lang_detector
 dotnet add package Panlingo.LanguageIdentification.CLD3 --version 0.0.0.18
 dotnet add package  Mosaik.Core --version 24.8.51117
 dotnet run
 dotnet lang_detector.dll
 ```
### License
https://creativecommons.org/licenses/by-sa/3.0/

### Links
https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes

```
@article{joulin2016bag,
  title={Bag of Tricks for Efficient Text Classification},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1607.01759},
  year={2016}
}

@article{joulin2016fasttext,
  title={FastText.zip: Compressing text classification models},
  author={Joulin, Armand and Grave, Edouard and Bojanowski, Piotr and Douze, Matthijs and J{\'e}gou, H{\'e}rve and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1612.03651},
  year={2016}
}

```
# Folder Florence_2
Contains Dotnet Inference scripts for Florence 2 in format ONNX

Download modesl from https://github.com/curiosity-ai/florence2-sharp/tree/main

# Folder mime

Conatains Mimetype detector in dotnet

### Requirements
- dotnet add package Mime-Detective --version 24.7.1
- dotnet add package Mime-Detective.Definitions.Exhaustive --version 24.7.1
- dotnet add package Mime-Detective.Definitions.Condensed --version 24.7.1


# Folder NSFW

Contains implementation of NoN Safe for Work Model in DotNet


 Run  pretrained  ONNX model using the Onnx Runtime C# API., the same we have in production in Python


### packages 
#### https://onnxruntime.ai/docs/api/csharp/api/Microsoft.ML.OnnxRuntime.html
- dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.0
- dotnet add package Microsoft.ML.OnnxRuntime.Managed --version 1.16.0
- dotnet add package Microsoft.ML
#### https://www.nuget.org/packages/SixLabors.ImageSharp
- dotnet add package SixLabors.ImageSharp --version 3.1.4


// Model fined tuned Vision Transformer https://huggingface.co/google/vit-base-patch16-224


# Folder OCR
Contains Implementation in DotNet of Tesseract OCR

Tesseract was originally developed at Hewlett-Packard Laboratories Bristol UK and at Hewlett-Packard Co, Greeley Colorado USA between 1985 and 1994, 
 * with some more changes made in 1996 to port to Windows, and some C++izing in 1998. In 2005 Tesseract was open sourced by HP. From 2006 until November 2018 
 * it was developed by Google.
 https://github.com/tesseract-ocr/tesseract
 

### Packages 
- dotnet add package TesseractOCR --version 5.3.5
- dotnet add package Spectre.Console --version 0.49.1

The DLL's Tesseract53.dll (and exe) and leptonica-1.83.0.dll are compiled with Visual Studio 2022 you need these C++ runtimes for it on your computer

- X86: https://aka.ms/vs/17/release/vc_redist.x86.exe
- X64: https://aka.ms/vs/17/release/vc_redist.x64.exe


MODELS https://github.com/tesseract-ocr/tessdata to folder tessdata


DOC https://tesseract-ocr.github.io/tessdoc/Command-Line-Usage.html


# Folder mlnet_image_classification

Image classification with tensorflow and MLNet

### Packages

- dotnet add package Microsoft.ML --version 4.0.0-preview.24378.1
- dotnet add package Microsoft.ML.ImageAnalytics --version 4.0.0-preview.24378.1
- dotnet add package Microsoft.ML.Vision --version 4.0.0-preview.24378.1
- dotnet add package SciSharp.TensorFlow.Redist-Windows-GPU --version 2.10.3
- dotnet add package Microsoft.ML.TensorFlow --version 4.0.0-preview.24378.1
-  dotnet add package Spectre.Console --version 0.49.1
### Dataset

Dataset https://digitalcommons.usu.edu/all_datasets/48/ SDNET2018: A concrete crack image dataset for machine learning applications
```
 Citation 
S. Dorafshan and M. Maguire, "Autonomous detection of concrete cracks on bridge decks and fatigue cracks on steel members," in Digital Imaging 2017, Mashantucket, CT, 2017. 
S. Dorafshan, M. Maguire and M. Chang, "Comparing automated image-based crack detection techniques in spatial and frequency domains," in Proceedings of the 26th American Society of Nondestructive Testing Research Symposium, Jacksonville, FL, 2017. 
S. Dorafshan, M. Maguire, N. Hoffer and C. Coopmans, "Challenges in bridge inspection using small unmanned aerial systems: Results and lessons learned," in Proceedings of the 2017 International Conference on Unmanned Aircraft Systems, Miami, FL, 2017. 
S. Dorafshan, C. Coopmans, R. J. Thomas and M. Maguire, "Deep Learning Neural Networks for sUAS-Assisted Structural Inspections, Feasibility and Application," in ICUAS 2018, Dallas, TX, 2018. 
S. Dorafshan, M. Maguire and X. Qi, "Automatic Surface Crack Detection in Concrete Structures Using OTSU Thresholding and Morphological Operations," Utah State University, Logan, Utah, USA, 2016.
S. Dorafshan, J. R. Thomas and M. Maguire, "Comparison of Deep Learning Convolutional Neural Networks and Edge Detectors for Image-Based Crack Detection in Concrete," Submitted to Journal of Construction and Building Materials, 2018. 
S. Dorafshan, R. Thomas and M. Maguire, "Image Processing Algorithms for Vision-based Crack Detection in Concrete Structures," Submitted to Advanced Concrete Technology, 2018.  
```

# Folder tf_image_classification

Tensoflow ML.Net and Image Classification. Imagenet Classification. 1000 Classes

- dotnet add package Microsoft.ML --version 3.0.1
- dotnet add package Microsoft.ML.TensorFlow --version 3.0.1
- dotnet add package Microsoft.ML.ImageAnalytics --version 3.0.1
- dotnet add package SciSharp.TensorFlow.Redist --version 2.16.0

 ### Description

 http://tersorflow.org
 TensorFlow is an open-source machine learning library developed by Google. TensorFlow is used to build and train deep learning models as it facilitates the creation of 
 computational graphs and efficient execution on various hardware platforms. The article provides an comprehensive overview of tensorflow. https://github.com/SciSharp/TensorFlow.NET  
 TensorFlow.NET (TF.NET) provides a .NET Standard binding for TensorFlow. It aims to implement the complete Tensorflow API in C# which allows .NET developers 
 to develop, train and deploy Machine Learning models with the cross-platform .NET Standard framework. TensorFlow.NET has built-in Keras high-level interface and 
 is released as an independent package TensorFlow.Keras.
 
 ML.Net https://dotnet.microsoft.com/en-us/apps/machinelearning-ai/ml-dotnet
 Microsoft.ML.TensorFlow  contains ML.NET integration of TensorFlow
 Microsoft.ML.ImageAnalytics work with Images
 