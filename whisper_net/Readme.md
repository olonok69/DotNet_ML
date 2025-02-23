# OpenAI Whisper
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



# Installation
Clone the repository:
```
git clone https://github.com/your-username/audio-transcription-api.git
```
Navigate to the project directory:
```
cd audio-transcription-api
```
Restore NuGet packages:
```
dotnet restore
```
Build the project:
```
dotnet build
```
# Running the API
```
dotnet run
```
The API will start listening on http://localhost:8899 (or the port you specified in Program.cs).

# Usage
Endpoint: /api/Audio/upload

Method: POST

Request Body:

audioFile: The audio file to transcribe (multipart/form-data).
translate (optional): Boolean value indicating whether to translate the transcription to English. Default is false.
Example using cURL:

```
curl -X POST -F audioFile=@path/to/your/audio.mp3 http://localhost:8899/api/Audio/upload
```
Response:
```
{
  "message": "Audio transcribed.",
  "transcription": "00:00.000->00:04.540: This is a test audio file.\n",
  "executionTimeMs": 1234
}
```
# Notes
- The API will automatically download the ggml-base.bin model file if it doesn't exist in the project directory.
- You can change the model type by modifying the ggmlType variable in the AudioController.cs file.
- The NAudio library is used for audio processing, including resampling.
- Make sure to adjust the API endpoint URL if you deploy the application to a different environment.
- You can use tools like Postman or cURL to send requests to the API.
- Refer to the api.cs file for the API implementation details.
- Refer to the Program.cs file for the application configuration.

- This README provides a basic guide. You may need to adjust it based on your specific needs and deployment environment.

# License

See Repository https://github.com/sandrohanea/whisper.net

Mit License https://github.com/sandrohanea/whisper.net/blob/main/LICENSE