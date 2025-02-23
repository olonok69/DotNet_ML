# Introduction 
Machine Learning Projects in DotNet
# Folder Semantic Kernel
Projects and Tests with Microsoft Semantic Kernel

### Folder phi3_sk
Integration of Microsoft Phi-3-mini-4k-instruct-onnx with Semantic Kernel Example

```
# Install Requirements DotNet
dotnet add package Microsoft.ML.OnnxRuntime --version 1.18.0
dotnet add package Spectre.Console --version 0.49.1 A .NET library that makes it easier to create beautiful, cross platform, console applications.
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.3.0  This API gives you an easy, flexible and performant way of running LLMs on device.
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --version 0.3.0
dotnet add package Microsoft.SemanticKernel
dotnet add package feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI.CPU or feiyun0112.SemanticKernel.Connectors.OnnxRuntimeGenAI.CUDA

# Look Example Semantic Kernel Connectors
https://github.com/feiyun0112/SemanticKernel.Connectors.OnnxRuntimeGenAI/tree/main

```

```
# Dependencies Python

## Download phi3 onnx
https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
https://onnxruntime.ai/docs/genai/tutorials/phi3-python.html

# pip install huggingface-hub[cli]

huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
```

### Folder sk1 plugins and prompts
Chat with Openai and Azure OpenAi Models using Semantic Kernel

### Folder OllamaSharp
Integrate Semantic Kernel with Ollama. Use Microsoft Phi3 model in local.

*Install*
You need to have an Ollama docker runing in local
```
dotnet add package Microsoft.SemanticKernel
// https://github.com/microsoft/SemanticKernelCookBook
dotnet add package OllamaSharp --version 2.0.10
// https://github.com/awaescher/OllamaSharp
// OLLAMA
docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
ollama run phi3 
```

### Folder native_function
create plugings , extend the functionality of semantic kernel. The Semantic Kernel SDK allows developers to run functions within prompts to create intelligent applications.
Functions nested within your prompts can perform a wide range of tasks to make your AI agent more robust.
This allows you to perform tasks that large language models can't typically complete on their own.
Using variables.
Calling external functions.
Passing arguments to functions.


```
# Dependencies DotNET
dotnet add package Microsoft.SemanticKernel


https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/adding-native-plugins?pivots=programming-language-csharp
https://learn.microsoft.com/en-us/semantic-kernel/concepts/plugins/?pivots=programming-language-csharp
```
 ### Folder dalle3
Dalle3 with Microsoft Semantic kernel
### Install
 - dotnet add package Microsoft.SemanticKernel --version 1.19.0

 https://github.com/microsoft/semantic-kernel/tree/main

  Semantic Kernel is an SDK that integrates Large Language Models (LLMs) like OpenAI, Azure OpenAI, and Hugging Face with conventional programming languages 
  like C#, Python, and Java. Semantic Kernel achieves this by allowing you to define plugins that can be chained together in just a few lines of code.
  
- dotnet add package System.Numerics.Tensors  --version 8.0.0

- dotnet add package  SkiaSharp --version 2.88.3

https://github.com/mono/SkiaSharp

SkiaSharp is a cross-platform 2D graphics API for .NET platforms based on Google's Skia Graphics Library (skia.org). 
It provides a comprehensive 2D API that can be used across mobile, server and desktop models to render images.

- dotnet add package Spectre.Console --version 0.49.1

Spectre.Console is a .NET library that makes it easier to create beautiful console applications.
https://spectreconsole.net/#:~:text=Spectre.Console.Cli.%20Create%20strongly%20typed%20settings%20and

- dotnet add package Spectre.Console.ImageSharp
### Folder planners
AI PLANNERS. Semantic Kernel SDK supports planners, which use artificial intelligence (AI) to automatically call the appropriate plugins for a given scenario.

```
# Dependencies DotNET
dotnet add package Microsoft.SemanticKernel.Planners.Handlebars --version 1.2.0-preview    NOTE PREVIEW
dotnet add package Microsoft.SemanticKernel
```

### Folder Chroma_app
 Semantic Kernel and Chroma Vector store
#### Chroma
https://docs.trychroma.com/
Chroma is an open-source embedding database designed to make it easy to build Language Model applications by making knowledge, facts, and plugins 
 * pluggable for LLMs. It allows us to store and retrieve information in a way that can be easily utilized by the models, enabling both short-term and long-term
 * memory for more advanced applications. 




```
# Docker Chroma
docker pull chromadb/chroma
docker run -it --rm -p 8000:8000/tcp chromadb/chroma:latest
 * */

# INSTALL Dependencies Dotnet
dotnet add package Microsoft.SemanticKernel
dotnet add package Microsoft.SemanticKernel.Plugins.Memory --version 1.16.2-alpha
dotnet add package Microsoft.SemanticKernel.Connectors.Chroma --version 1.16.2-alpha
dotnet add package System.Linq.Async
```

# Folder phi3Vision
Inference Microsoft Phi3 Vision in ONNX format in CPU
Phi-3-vision-128k-instruct allows Phi-3 to not only understand language, but also see the world visually. Through Phi-3-vision-128k-instruct,
we can solve different visual problems, such as OCR, table analysis, object recognition, describe the picture etc.
We can easily complete tasks that previously required a lot of data training. The following are related techniques and application scenarios cited by
Phi-3-vision-128k-instruct.

```
# Install Dependencies DotNet
dotnet add package Spectre.Console --version 0.49.1 A .NET library that makes it easier to create beautiful, cross platform, console applications.
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.3.0  This API gives you an easy, flexible and performant way of running LLMs on device.

```
```
# Dependencies python

 pip install huggingface-hub[cli]
huggingface-cli download microsoft/Phi-3-vision-128k-instruct-onnx-cpu --include cpu-int4-rtn-block-32-acc-level-4/* --local-dir .
```
#### ONNX

https://onnxruntime.ai/docs/genai/tutorials/phi3-python.html

# Folder ai_toolkit
The AI Toolkit for Visual Studio Code (VS Code) is a VS Code extension that simplifies generative AI app development by bringing together cutting-edge AI development tools and models from the Azure AI Foundry catalog and other catalogs like Hugging Face

Example Inference Phi3 quantized 4int onnx in Local
```
# Dotnet Dependencies
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.DirectML --version 0.3.0
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --version 0.3.0
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.3.0
```

# Folder Phi3
Inference with Microsoft Phi3 in ONNX format in CPU
```
# Install Dependencies DotNet
dotnet add package Microsoft.ML.OnnxRuntime --version 1.18.0
dotnet add package Microsoft.ML.OnnxRuntimeGenAI --version 0.3.0-rc2
dotnet add package Microsoft.ML.OnnxRuntimeGenAI.Cuda --version 0.3.0-rc2

# Download phi3 onnx
// https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-onnx
// https://onnxruntime.ai/docs/genai/tutorials/phi3-python.html
// pip install huggingface-hub[cli]

// huggingface-cli download microsoft/Phi-3-mini-4k-instruct-onnx --include cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/* --local-dir .

```

# Folder bert_onnx
Inference with Bert Model in ONNX format
### Install
```
// packages 
dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.0
dotnet add package Microsoft.ML.OnnxRuntime.Managed --version 1.16.0
dotnet add package Microsoft.ML
dotnet add package BERTTokenizers --version 1.1.0
```
You also need to download the ONNX model and adjust the modelPath variable. You can download the model from Huggingface Gub
https://huggingface.co/google-bert/bert-large-uncased-whole-word-masking-finetuned-squad and convert to onnx

var modelPath = @"d:\repos\onnx\models\bert-large-uncased-whole-word-masking-finetuned-squad-17.onnx";

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


# Folder tesseract_dotnet
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
 

 # Folder dalle3
Dalle3 with Microsoft Semantic kernel
### Install
 - dotnet add package Microsoft.SemanticKernel --version 1.19.0

 https://github.com/microsoft/semantic-kernel/tree/main

  Semantic Kernel is an SDK that integrates Large Language Models (LLMs) like OpenAI, Azure OpenAI, and Hugging Face with conventional programming languages 
  like C#, Python, and Java. Semantic Kernel achieves this by allowing you to define plugins that can be chained together in just a few lines of code.
  
- dotnet add package System.Numerics.Tensors  --version 8.0.0

- dotnet add package  SkiaSharp --version 2.88.3

https://github.com/mono/SkiaSharp

SkiaSharp is a cross-platform 2D graphics API for .NET platforms based on Google's Skia Graphics Library (skia.org). 
It provides a comprehensive 2D API that can be used across mobile, server and desktop models to render images.

- dotnet add package Spectre.Console --version 0.49.1

Spectre.Console is a .NET library that makes it easier to create beautiful console applications.
https://spectreconsole.net/#:~:text=Spectre.Console.Cli.%20Create%20strongly%20typed%20settings%20and

- dotnet add package Spectre.Console.ImageSharp

# Folder torchsharp
Use of Torch in DotNet with MLNet


 ### ML.NET 
 
 Is an open-source, cross-platform machine learning framework for .NET developers that enables integration of custom machine learning models into .NET applications. 
 It encompasses an API, which consists of different NuGet packages, a Visual Studio extension called Model Builder, and a command-line interface that's installed as a .NET tool.
 
 #### Features
 
 * Custom ML made easy with AutoML
   ML.NET offers Model Builder (a simple UI tool) and ML.NET CLI to make it super easy to build custom ML Models.
   These tools use Automated ML (AutoML), a cutting edge technology that automates the process of building best performing models for your Machine Learning scenario. 
   All you have to do is load your data, and AutoML takes care of the rest of the model building process.

 * Extended with TensorFlow & more
   ML.NET has been designed as an extensible platform so that you can consume other popular ML frameworks (TensorFlow, ONNX, Infer.NET, and more) and have access 
   to even more machine learning scenarios, like image classification, object detection, and more.

 * High performance and accuracy

 
### Install 
dotnet add package Microsoft.ML.TorchSharp --version 0.22.0-preview.24378.1

# Folder Object Detection
This model is a real-time neural network for object detection that detects 80 different Objects
Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks. https://arxiv.org/pdf/1506.01497
## Open Neural Network Exchange ONNX
Models --> https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/faster-rcnn#preprocessing-steps
```
# Requirements DotNet
dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.3
dotnet add package SixLabors.ImageSharp --version 2.1.8
dotnet add package SixLabors.ImageSharp.Drawing --version 1.0.0-beta14
```
# Folder Extract Images
Extract Images from Documents with GroupDocs Parser
- https://dev.to/usmanaziz/extract-images-from-pdf-documents-using-c-net-207n
- https://products.groupdocs.com/parser/
- https://docs.groupdocs.com/parser/net/system-requirements/


# Folder ImageClassification

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


### Folder whisper_net

Transcription and Translation using OpenAi Whisper

OpenAI Whisper is a general-purpose speech recognition model developed by OpenAI.

 It was trained on a large dataset of diverse audio and is also a multi-task model that can perform multilingual speech recognition, speech translation, and language identification.
# Audio Transcription API with Whisper.net
This is a simple ASP.NET Core Web API for transcribing audio files using the Whisper.net library, which provides a C# implementation of OpenAI's Whisper speech recognition model.

# Features
Transcribes audio files in MP3, WAV, and x-WAV formats.
Automatically downloads the Whisper.net model if it's not found.
Resamples audio to 16kHz if necessary.
Option to translate the transcribed text to English.
# Dependencies
*.NET 8 SDK

- Whisper.net NuGet package
- "Whisper.net.AllRuntimes" Version="1.7.4"
- NAudio NuGet package
- "Microsoft.AspNetCore.Mvc.Core" Version="2.3.0"
- "Microsoft.AspNetCore.Server.Kestrel" Version="2.3.0"
- "Microsoft.VisualStudio.Azure.Containers.Tools.Targets" Version="1.21.0"