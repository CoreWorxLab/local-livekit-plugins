import cv2
import numpy as np
import os
import time

def test_display(display):
    print(f"Testing DISPLAY={display}")
    os.environ["DISPLAY"] = display
    
    # Try to create a simple window
    img = np.zeros((480, 640, 3), np.uint8)
    cv2.putText(img, f"Testing Display {display}", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    window_name = f"Test Window {display}"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        key = cv2.waitKey(2000)
        print(f"Window shown on {display}, waitKey returned {key}")
        cv2.destroyWindow(window_name)
        return True
    except Exception as e:
        print(f"Failed on {display}: {str(e)}")
        return False

if __name__ == "__main__":
    displays = [":0", ":1", ":2"]
    current = os.environ.get("DISPLAY", "None")
    print(f"Current DISPLAY in environment: {current}")
    
    for d in displays:
        test_display(d)
