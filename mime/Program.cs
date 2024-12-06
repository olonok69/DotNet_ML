//dotnet add package Mime-Detective --version 24.7.1
//dotnet add package Mime-Detective.Definitions.Exhaustive --version 24.7.1
// dotnet add package Mime-Detective.Definitions.Condensed --version 24.7.1
using MimeDetective;
using System;
using System.Threading.Tasks;

using MimeDetective.Definitions;

class Program
 {
        private static void Main(string[] args)

    {
        var Inspector = new ContentInspectorBuilder()
        {
            Definitions = new MimeDetective.Definitions.ExhaustiveBuilder()
            {
                UsageType = MimeDetective.Definitions.Licensing.UsageType.PersonalNonCommercial
            }.Build()
        }.Build();


            var ContentFileName = @"D:\repos2\c#\mimetype\mime\data\example.png";

            var Results = Inspector.Inspect(ContentFileName);

            var ResultsByFileExtension = Results.ByFileExtension();
            var ResultsByMimeType = Results.ByMimeType();
            Console.WriteLine(ResultsByFileExtension[0].Extension);
        Console.WriteLine(ResultsByMimeType[0].MimeType);
        }
  }

