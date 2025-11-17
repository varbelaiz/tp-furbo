import cv2

input_video = '1_720p.mkv'
output_video = 'mediocampo.mp4'

cap = cv2.VideoCapture(input_video)

# 1. Read the FPS first so the frame indices can be computed
fps = cap.get(cv2.CAP_PROP_FPS) 
if fps == 0: fps = 30 # Fallback for the rare case where FPS is not reported
fps = int(fps)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 2. Define the cut points in seconds
start_seconds = (0 * 60) + 00
end_seconds = (0 * 60) + 8

# 3. Convert them into exact frame indices
start_frame = int(start_seconds * fps)
end_frame = int(end_seconds * fps)
duration_frames = end_frame - start_frame

print(f"Video a {fps} FPS.")
print(f"Cortando desde {start_seconds}s (Frame {start_frame}) hasta {end_seconds}s (Frame {end_frame})")

out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Seek to the first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

current_frame = start_frame

while cap.isOpened() and current_frame < end_frame:
    ret, frame = cap.read()
    if not ret:
        break
    out.write(frame)
    current_frame += 1
    
    # Simple progress indicator
    if (current_frame - start_frame) % 30 == 0:
        print(f"Procesado: {current_frame - start_frame}/{duration_frames} frames", end='\r')

cap.release()
out.release()
print(f"\nDone. Written to {output_video}")