// dotnet add package Panlingo.LanguageIdentification.CLD3 --version 0.0.0.18
// https://github.com/gluschenko/language-identification/tree/master

/* Install
 * 
 * sudo apt -y install protobuf-compiler libprotobuf-dev nuget
 * 
 * wget https://packages.microsoft.com/config/debian/12/packages-microsoft-prod.deb -O packages-microsoft-prod.deb
 * sudo dpkg -i packages-microsoft-prod.deb
 * rm packages-microsoft-prod.de
 * sudo apt-get update &&   sudo apt-get install -y dotnet-sdk-8.0
 * 
 * build app in LInux
 * dotnet new create console -n lang_detector
 *  dotnet add package Panlingo.LanguageIdentification.CLD3 --version 0.0.0.18
 *  dotnet add package  Mosaik.Core --version 24.8.51117
 *  dotnet run
 *  dotnet lang_detector.dll
 *  
 * https://creativecommons.org/licenses/by-sa/3.0/
 * https://en.wikipedia.org/wiki/List_of_ISO_639_language_codes
*
 * @article{joulin2016bag,
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

 */

using Panlingo.LanguageIdentification.CLD3;
namespace languages.LanguageDetection;
class Program
{
    static void Main()
    {
        // Create an instance of the language detector
        using var cld3 = new CLD3Detector(minNumBytes: 0, maxNumBytes: 512);

        //var text = "Hello, how are you? Привіт, як справи? Привет, как дела?";

        var singlePrediction = cld3.PredictLanguage("Привіт, як справи?");

        Console.WriteLine($"Language: {singlePrediction.Language}");
        Console.WriteLine($"Probability: {singlePrediction.Probability}");
        Console.WriteLine($"IsReliable: {singlePrediction.IsReliable}");
        Console.WriteLine($"Proportion: {singlePrediction.Proportion}");

        var predictions = cld3.PredictLanguages("Hello, how are you? Привіт, як справи? Привет, как дела?", 3);

        foreach (var prediction in predictions)
        {
            Console.WriteLine(
                $"Language: {prediction.Language}, " +
                $"Probability: {prediction.Probability}, " +
                $"IsReliable: {prediction.IsReliable}, " +
                $"Proportion: {prediction.Proportion}"
            );
        }

        foreach (var (lang, texto) in Data.ShortSamples)
        {
            var predictions2 = cld3.PredictLanguages(texto, 1);
            foreach (var prediction in predictions2)
            {
                Console.WriteLine(
                    $"Language_original: {lang}, " +
                    $"Language_Detectado: {prediction.Language}, " +
                    $"Probability: {prediction.Probability}, " +
                    $"IsReliable: {prediction.IsReliable}, " +
                    $"Proportion: {prediction.Proportion}"
                );
            }
        }
    }
}