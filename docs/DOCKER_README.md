# Docker Deployment Guide

## Prerequisites
- Docker installed on your system
- Docker Compose (optional, but recommended)

## Build and Run

### Option 1: Using Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up --build

# Or run in detached mode
docker-compose up -d --build

# Stop the container
docker-compose down
```

### Option 2: Using Docker CLI

```bash
# Build the image
docker build -t emotion-audio-app .

# Run the container
docker run -p 7860:7860 emotion-audio-app

# Or run in detached mode
docker run -d -p 7860:7860 --name emotion-app emotion-audio-app
```

## Access the Application

Once running, open your browser and navigate to:
```
http://localhost:7860
```

## GPU Support

To enable GPU support (if you have NVIDIA GPU and nvidia-docker installed):

1. Uncomment the GPU section in `docker-compose.yml`
2. Or for Docker CLI:
```bash
docker run --gpus all -p 7860:7860 emotion-audio-app
```

## Updating Models

Models are copied into the Docker image during build. To update models:

1. Replace the model files in the `models/` directory
2. Rebuild the image:
   ```bash
   docker-compose up --build
   ```

Alternatively, you can use volume mounting (already configured in docker-compose.yml) to update models without rebuilding.

## Troubleshooting

### Port already in use
If port 7860 is already in use, change the port mapping:
```bash
docker run -p 8080:7860 emotion-audio-app
```
Then access via `http://localhost:8080`

### Memory issues
If you encounter memory issues, increase Docker's memory allocation in Docker Desktop settings.
