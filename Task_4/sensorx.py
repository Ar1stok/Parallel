import os
import time
import queue
import logging
import argparse
import threading
import cv2

stop_event = threading.Event()

class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    def __init__(self, delay: float) -> None:
        self._delay = delay
        self._data = 0
        
    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data
    

class SensorCam(Sensor):
    def __init__(self, device, width, height):
        self.device = device
        self.cap = cv2.VideoCapture(self.device)

        if not self.cap.isOpened():
            logging.error("Camera index out of range")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def get(self):
        ret, frame = self.cap.read()
        if not ret:
            logging.error("The frame was not received")
            stop_event.set()
            exit()
        return frame

    def __del__(self):
        self.cap.release()

  
class ShowImg(Sensor):
    def __init__(self, delay=0.01):
        if delay < 0.01: 
            self._delay_img = 0.01
            logging.info("The delay was set by default = 0.01s")
        else: 
            self._delay_img = delay
        
        self._last_value = [0, 0, 0]
  
    def show(self, frame: cv2.typing.MatLike, data_queue: list[queue.Queue]):    
        time.sleep(self._delay_img)
        
        for i in range(3):
            if not data_queue[i].empty():
                self._last_value[i] = data_queue[i].get()
        
        cv2.rectangle(frame, (frame.shape[1] - 480, frame.shape[0] - 25), 
                        (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)
        
        cv2.putText(frame, f'Sensor_1 = {self._last_value[0]}, Sensor_2 = {self._last_value[1]}, Sensor_3 = {self._last_value[2]}', 
                        (frame.shape[1] - 470, frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.imshow("Frame", frame)

    def __del__(self):
        cv2.destroyAllWindows()


def Parser():
    parser = argparse.ArgumentParser(description='Video options')
    parser.add_argument('--cam_id', type=int, default=0, help='Camera id')
    parser.add_argument('--width', type=int, default=640, help='Width of resolution')
    parser.add_argument('--height', type=int, default=480, help='Height of resolution')
    parser.add_argument('--delay', type=float, default=0.01, help='Delay for cam')
    parser.add_argument('--delay_s1', type=float, default=0.01, help='Delay for s1')
    parser.add_argument('--delay_s2', type=float, default=0.1, help='Delay for s2')
    parser.add_argument('--delay_s3', type=float, default=1, help='Delay for s3')
    return parser.parse_args()


def work_put(sensor: SensorX, data_queue: queue.Queue):
    while not stop_event.is_set():
        data = sensor.get()
        if data_queue.full():
            try:
                data_queue.get_nowait()
            except queue.Empty:
                pass   
        data_queue.put(data)
            

def work_get(cam_q: queue.Queue):
    frame = cam_q.get()
    return frame


if __name__ == "__main__":
    if not os.path.exists('log'):
        os.makedirs('log')
    
    log_file = os.path.join('log','Info.log')
    logging.basicConfig(filename=log_file, level=logging.INFO, filemode='w', 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    args = Parser()
    cam_q = queue.Queue(maxsize=1)
    data_queue = [queue.Queue(maxsize=1) for _ in range(3)]
    
    cam = SensorCam(args.cam_id, args.width, args.height)
    sensor0 = SensorX(args.delay_s1)
    sensor1 = SensorX(args.delay_s2)
    sensor2 = SensorX(args.delay_s3)
    image = ShowImg(args.delay)
    
    threading.Thread(target=work_put, args=(cam, cam_q)).start()
    threading.Thread(target=work_put, args=(sensor0, data_queue[0])).start()
    threading.Thread(target=work_put, args=(sensor1, data_queue[1])).start()
    threading.Thread(target=work_put, args=(sensor2, data_queue[2])).start()
    
    while True:
        frame = work_get(cam_q)
        image.show(frame, data_queue)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            stop_event.set()
            break

    del cam
    del image
    
    logging.info("Work is done")