import os

dataset_path = "D:\\semester4\\computer_vision\\project\\emotiondataset\\train"

emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Count images in each folder
for emotion in emotions:
    folder_path = os.path.join(dataset_path, emotion)
    num_images = len(os.listdir(folder_path))
    print(f"{emotion.capitalize()}: {num_images} images")
