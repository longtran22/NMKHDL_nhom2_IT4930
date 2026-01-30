# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY app.py .
COPY model.py .
COPY model_image.py .
COPY inference.py .

# Create models directories
RUN mkdir -p models/images

# Copy models from Emotion_audio/models directory
COPY models/best_model.pth models/
COPY models/images/best_resnet18_finetune4.pth models/images/

# Expose port for Gradio
EXPOSE 7860

# Set environment variable for Gradio
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# Run the application
CMD ["python", "app.py"]
