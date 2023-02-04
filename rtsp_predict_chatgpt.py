import cv2
import numpy as np
import time
import darknet

def get_frame(cap):
    ret, frame = cap.read()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return None

def display_results(frame, results, classes):
    for cat, score, bounds in results:
        x, y, w, h = bounds
        cv2.rectangle(frame, (int(x-w/2),int(y-h/2)), (int(x+w/2),int(y+h/2)), (255,0,0))
        cv2.putText(frame, str(classes[int(cat)]) + " [" + str(score) + "]", (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 2)
    cv2.imshow("Object Detection", frame)

def get_test_input(input_dim, CUDA):
    img = cv2.imread("imgs/messi.jpg")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    
    if CUDA:
        img_ = img_.cuda()
    
    return img_


if __name__ == "__main__":
    cap = cv2.VideoCapture("rtsp://[camera_url]")

    net = darknet.load_net(b"cfg/yolov3.cfg", b"weights/yolov3.weights", 0)
    classes = darknet.load_classes(b"data/coco.names")


    while True:
        frame = get_frame(cap)
        if frame is None:
            break

        sized = cv2.resize(frame, (darknet.network_width(net), darktet.network_height(net)))
        darknet_image = darknet.ndarray_to_image(sized)
        results = darknet.detect_image(net, classes, darknet_image, thresh=0.5)

        display_results(frame, results, classes)

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
