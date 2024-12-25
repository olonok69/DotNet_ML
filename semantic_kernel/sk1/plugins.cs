
//dotnet add package Microsoft.SemanticKernel.Plugins.Core --version 1.2.0-alpha
using Microsoft.SemanticKernel;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using Microsoft.SemanticKernel.Plugins.Core;
using System;

var builder = Kernel.CreateBuilder();

var azureopenaikey = Environment.GetEnvironmentVariable("AZURE_OPENAI_API_KEY");

builder.AddAzureOpenAIChatCompletion(
         "gpt-3-5-16k",                      // Azure OpenAI Deployment Name
         "https://open-ai-olonok.openai.azure.com", // Azure OpenAI Endpoint
         azureopenaikey);


builder.Plugins.AddFromType<TimePlugin>();
var kernel = builder.Build();
var currentDay = await kernel.InvokeAsync("TimePlugin", "DayOfWeek");
Console.WriteLine(currentDay);