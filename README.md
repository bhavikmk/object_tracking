# Leaves Disease Detection Module

This module is designed to detect plant leaf diseases in real-time using computer vision and machine learning. It leverages a pre-trained deep learning model to classify leaf health from live video feeds, identifying whether a leaf is healthy or infected with specific diseases.

### Features
- Real-time Leaf Disease Detection: Process video frames from a webcam or camera feed in real-time.
- Disease Classification: Detect various leaf diseases (e.g., Powdery Mildew, Leaf Spot, Yellow Mosaic, Rust) or classify as healthy.
- Confidence Scores: Display prediction confidence for disease classification.
- Customizable Model: The module can be easily updated with new models trained on different plant species or diseases.

### Further Enhancements
- Disease Segmentation: Implement segmentation techniques (e.g., U-Net) to precisely identify the infected regions of the leaf.
- Mobile Deployment: Convert the trained model to TensorFlow Lite for real-time mobile use (iOS/Android).
- Large-Scale Monitoring: Integrate the system with drones or IoT devices for monitoring large-scale farms or gardens.