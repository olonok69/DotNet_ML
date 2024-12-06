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
