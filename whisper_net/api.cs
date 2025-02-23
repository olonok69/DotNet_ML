using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using NAudio.Wave;
using Whisper.net;
using Whisper.net.Ggml;


namespace AudioUpload.Controllers
{
    [ApiController]
    [Route("api/[controller]")]
    public class AudioController : ControllerBase
    {
        private readonly string _modelPath; // Store the model path
        private readonly IWebHostEnvironment _env;

        // Constructor to inject the model path (best practice)
        public AudioController(IWebHostEnvironment env) // Inject IWebHostEnvironment
        {
            _env = env;
            // Example: Model in wwwroot folder. Adjust path as needed.
            _modelPath = Path.Combine(env.ContentRootPath, "ggml-base.bin");

            // Check if the model file exists (important!)
            if (!System.IO.File.Exists(_modelPath))
            {
                DownloadModelIfNeeded().Wait(); // Download if needed.
            }
        }

        private async Task DownloadModelIfNeeded()
        {
            if (!System.IO.File.Exists(_modelPath))
            {
                var ggmlType = GgmlType.Base;
                var modelFileName = "ggml-base.bin";
                await DownloadModel(modelFileName, ggmlType);
            }
        }

        private async Task DownloadModel(string modelFileName, GgmlType ggmlType)
        {
            try
            {
                var modelUri = ggmlType switch
                {
                    GgmlType.Tiny => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
                    GgmlType.Base => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
                    GgmlType.Small => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
                    GgmlType.Medium => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
                    GgmlType.LargeV1 => "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin",
                    _ => throw new ArgumentOutOfRangeException(nameof(ggmlType), ggmlType, null)
                };

                using var httpClient = new HttpClient();
                var response = await httpClient.GetAsync(modelUri, HttpCompletionOption.ResponseHeadersRead);
                response.EnsureSuccessStatusCode();

                using var contentStream = await response.Content.ReadAsStreamAsync();
                using var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.None, 8192, true);
                await contentStream.CopyToAsync(fileStream);

                Console.WriteLine($"Downloaded model to {_modelPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error downloading model: {ex.Message}");
                throw; // Re-throw to indicate download failure.
            }
        }



        [HttpPost("upload")]
        public async Task<IActionResult> UploadAudio(IFormFile audioFile, bool translate = false)
        {

            audioFile.OpenReadStream().Position = 0; // Reset stream position!

            if (audioFile == null || audioFile.Length == 0)
            {
                return BadRequest("No audio file received.");
            }

            if (audioFile.ContentType != "audio/mpeg" &&
                audioFile.ContentType != "audio/wav" &&
                audioFile.ContentType != "audio/x-wav")
            {
                return BadRequest("Invalid audio file format. Only MP3, WAV and x-WAV files are allowed.");
            }

            try
            {
                // 1. Generate a temporary file path
                string tempFilePath = Path.GetTempFileName();
                Stopwatch stopwatch = Stopwatch.StartNew();


                // 2. Save the file to the temporary location
                using (var stream = new FileStream(tempFilePath, FileMode.Create))
                {
                    await audioFile.CopyToAsync(stream);
                }


                using (var reader = new AudioFileReader(tempFilePath)) // Use tempFilePath here
                {
                    if (reader.WaveFormat.SampleRate!= 16000)
                    {
                        // Resample the audio
                        using (var resampler = new MediaFoundationResampler(reader, new WaveFormat(16000, reader.WaveFormat.Channels)))
                        {
                            using (var memoryStream = new MemoryStream())
                            {
                                // Use WaveFileWriter to write to the memory stream
                                WaveFileWriter.WriteWavFileToStream(memoryStream, resampler);
                                memoryStream.Position = 0;

                                string transcribedText = await TranscribeAudioWithWhisper(memoryStream, translate);
                                stopwatch.Stop();
                                long executionTimeMs = stopwatch.ElapsedMilliseconds;

                                return Ok(new
                                {
                                    Message = "Audio transcribed.",
                                    Transcription = transcribedText,
                                    ExecutionTimeMs = executionTimeMs
                                });
                            }
                        }
                    }
                    else
                    {
                        // Audio is already 16kHz
                        string transcribedText = await TranscribeAudioWithWhisper(audioFile.OpenReadStream(), translate); // Use original stream here

                        stopwatch.Stop();
                        long executionTimeMs = stopwatch.ElapsedMilliseconds;

                        return Ok(new
                        {
                            Message = "Audio transcribed.",
                            Transcription = transcribedText,
                            ExecutionTimeMs = executionTimeMs
                        });
                    }
                }
            }
            catch (InvalidDataException ex)
            {
                return BadRequest($"Invalid audio data: {ex.Message}");
            }
            catch (Exception ex)
            {
                return StatusCode(500, $"Internal server error: {ex.Message}");
            }
        }

        private async Task<string> TranscribeAudioWithWhisper(Stream audioStream,  bool translate)
        {
            var whisperFactory = WhisperFactory.FromPath(_modelPath); // Use the stored path
            var builder = whisperFactory.CreateBuilder()
                .WithLanguage("auto");

            if (translate)
            {
                builder = builder.WithTranslate();
            }

            using var processor = builder.Build();

            var transcription = "";
            await foreach (var result in processor.ProcessAsync(audioStream))
            {
                transcription += $"{result.Start}->{result.End}: {result.Text}\n";
            }

            return transcription;
        }
    }
}