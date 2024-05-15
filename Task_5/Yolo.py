import threading
import time
import logging
import argparse
import cv2
from ultralytics import YOLO

class Video:
    def __init__(self, video_path, out_name):
        self.path = video_path
        self.out = out_name
        self.frames_list = []
        self.cap = cv2.VideoCapture(self.path)
        
        if not self.cap.isOpened():
            logging.error("Video not found")
            exit()
            
        width, height, fps = (int(self.cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

        self.video_writer = cv2.VideoWriter(self.out,
                                            cv2.VideoWriter_fourcc(*'mp4v'),
                                            fps, (width, height))
    
    def get_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logging.info("Video frame is empty or video processing has been successfully completed.")
                break
            self.frames_list.append(frame)
            
    def save_video(self):
        for frame in self.frames_list:
            self.video_writer.write(frame)
        
    def __del__(self):
        cv2.destroyAllWindows()
        self.video_writer.release()
        

def processing(img_list: list, start: int, end: int) -> list:
    model = YOLO("yolov8s-pose.pt")
    for cnt in range(start, end):
        result = model.predict(img_list[cnt], conf=0.3, verbose=False)
        img_list[cnt] = result[0].plot()


def Parser():
    parser = argparse.ArgumentParser(description='Video options')
    parser.add_argument('--video_path', type=str, default="Dance.mp4", help='Video path')
    parser.add_argument('--num_thread', type=int, default=1, help='Number of threads (no more than 10)')
    parser.add_argument('--out', type=str, default="Result.avi", help='Output file name')
    return parser.parse_args()


if __name__ == "__main__":
    args = Parser()
    logging.basicConfig( filename="Info.log", level=logging.INFO, filemode='w', 
                         format='%(asctime)s - %(levelname)s - %(message)s' )
    video = Video(args.video_path, args.out)
    
    video.get_frames()
    
    threads = []
    if (args.num_thread > 10 or args.num_thread < 1):
        logging.warning("Num_thread set by default")
        num_thread = 1
    else: num_thread = args.num_thread
    frames_per_thread = len(video.frames_list) // num_thread
    print(frames_per_thread)
    
    start = time.perf_counter()
    for i in range(num_thread):
        if i == (num_thread - 1):
            thr = threading.Thread(target=processing, args=(video.frames_list, (frames_per_thread * i), len(video.frames_list)))
            thr.start()
            threads.append(thr)
        else:
            thr = threading.Thread(target=processing, args=(video.frames_list, (frames_per_thread * i), (frames_per_thread * (i + 1))))
            thr.start()
            threads.append(thr)
        
    for thr in threads:
        thr.join()
    end = time.perf_counter() - start
    logging.info(f"Processing is ended\nTime: {end:0.3f} second")
    
    video.save_video()
    
    del video