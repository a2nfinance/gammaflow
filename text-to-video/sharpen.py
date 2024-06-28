import torch
#from basicsr.archs.videovsr import BasicVSRPlusPlus
import cv2
import  mmcv.ops
from vsbasicvsrpp import BasicVSRPP


ret = BasicVSRPP("fake-jack.mp4")
print(ret)
# Load a pre-trained model
model = BasicVSRPlusPlus(in_nc=3, out_nc=3, nf=64, groups=8, n_frames=5)

# Load weights into the model
checkpoint = torch.load('spynet_20210409-c6c1bd09.pth', map_location='cpu')
model.load_state_dict(checkpoint['params'])

# Set the model to evaluation mode
model.eval()

# Assuming you have a video input 'input.mp4', and you want to process it frame by frame
# Use ffmpeg or another library to extract frames from the video
# Here's a simplified example using OpenCV to read frames



video_path = 'fake-jack.mp4'
output_path = 'output.mp4'

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to tensor and process with BasicVSR++
    input_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        output_tensor = model(input_tensor)

    # Convert back to numpy array and write to output video
    out_frame = (output_tensor.squeeze(0).clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    out.write(out_frame)

cap.release()
out.release()
cv2.destroyAllWindows()





# Initialize the model
model = RealESRGANer(model_path='RealESRGAN_x4plus.pth', scale=4, model = None)

# Load the input video
cap = cv2.VideoCapture('fake-jack.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('fake-jack_sharpened_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    # Enhance the frame using Real-ESRGAN
    enhanced_frame = model.enhance(frame)
    out.write(enhanced_frame)

cap.release()
out.release()
cv2.destroyAllWindows()