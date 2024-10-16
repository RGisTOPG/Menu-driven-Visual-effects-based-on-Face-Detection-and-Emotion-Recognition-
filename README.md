# Menu-driven-Visual-effects-based-on-Face-Detection-and-Emotion-Recognition-
This code captures video from the webcam, detects faces, and applies various effects based on user input or detected emotions. Effects include overlaying dog noses/faces, blurring frames, and playing emotion-specific background videos. Users can save frames with effects by pressing 7.

Features --
1. Real-Time Facial Recognition
    Face Detection: Utilizes OpenCV’s Haar cascades to detect faces in real-time from a webcam feed.
    Emotion Recognition: Integrates the FER (Facial Expression Recognition) model to identify emotions (e.g., happy, sad, angry) from detected faces, with                                real-time emotion tracking.
2. Interactive Visual Effects
   Dog Nose and Face Overlays: Adds custom transparent PNG overlays (e.g., dog nose, tongue) to detected face regions, positioning and blending them using                                     alpha channels.
   Video Backgrounds: Based on the detected emotion, the project dynamically loads and plays corresponding background videos (e.g., happy, sad), blending the                         video with the live webcam feed for subtle effects.
3. Advanced Image Processing
    Blurring Effects: Offers options to blur the entire frame or selectively blur everything except detected faces, keeping the subject in focus while applying                       creative background effects.
    Alpha Blending: Uses alpha blending techniques for seamless overlay of transparent images onto video frames, ensuring natural transitions between layers.
    Real-Time Feedback: Displays detected emotion labels and confidence scores on the live video feed, providing immediate feedback for each detected face.
4. User Interaction and Control
    Effect Switching: Users can toggle between different effects (e.g., dog nose, dog face, blur effects) through simple keypresses (1-7), allowing for                               interactive and customizable video manipulation.
    Frame Capture: The ability to save frames with applied effects to a designated folder, enabling easy capture and storage of images from live video.
5. Emotion-Based Dynamic Responses
    Emotion-Triggered Video Playback: Plays specific videos in the background based on detected facial emotions, looping seamlessly and adjusting to emotion                                          changes in real-time.
    Looping Video Playback: The project ensures smooth looping of background videos by resetting the video frames when the end                                                               is reached.
