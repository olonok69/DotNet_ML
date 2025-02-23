FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
RUN apt-get update && apt-get install -y ffmpeg wine
RUN winecfg  # Create Wine configuration
RUN wget https://raw.githubusercontent.com/z0z0z/mf-install/master/mf-install.sh
RUN bash mf-install.sh
WORKDIR /App

# Copy everything
COPY . ./
# Restore as distinct layers
RUN dotnet restore
# Build and publish a release
RUN dotnet build -c Release -o /App/build

# Build runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0
RUN apt-get update && apt-get install -y ffmpeg wine
RUN winecfg  # Create Wine configuration
RUN wget https://raw.githubusercontent.com/z0z0z/mf-install/master/mf-install.sh
RUN bash mf-install.sh
WORKDIR /App
COPY --from=build /App/build .

# Expose the port the application listens on
EXPOSE 8899

# Copy the Whisper model (ggml-base.bin) into the image. Place it in the wwwroot folder
COPY ggml-base.bin ./wwwroot/
COPY ggml-base.bin .

ENTRYPOINT ["dotnet", "AudioUpload.dll"]