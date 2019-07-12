import cv2
import numpy as np

image_hsv = None   # global ;(
pixel = (20,60,80) # some stupid default

# mouse callback function
def pick_color(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixel = image_hsv[y,x]
        range = image_hsv[1572:3458, 3826:4808]

        #you might want to adjust the ranges(+-10, etc):
        upper =  np.array([pixel[0] + 40, pixel[1] + 40, pixel[2] + 40])
        lower =  np.array([pixel[0] - 40, pixel[1] - 40, pixel[2] - 40])
        print(pixel, lower, upper)
        # print x, y
        image_mask = cv2.inRange(range,lower,upper)
        cv2.namedWindow('mask',cv2.WINDOW_NORMAL)
        cv2.resizeWindow('mask', 600,600)
        cv2.imshow("mask",image_mask)

def main():
    import sys
    global image_hsv, pixel # so we can use it in mouse callback

    image_hsv = cv2.imread(sys.argv[1])  # pick.py my.png
    if image_hsv is None:
        print ("the image read is None............")
        return
    cv2.namedWindow('bgr',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('bgr', 600,600)
    cv2.setMouseCallback('bgr', pick_color)
    cv2.imshow("bgr",image_hsv)

    ## NEW ##
    
    cv2.setMouseCallback('bgr', pick_color)

    # now click into the hsv img , and look at values:
    #image_hsv = cv2.cvtColor(image_src,cv2.COLOR_BGR2HSV)
    cv2.namedWindow('bgr',cv2.WINDOW_NORMAL)
    cv2.resizeWindow('bgr', 600,600)
    cv2.imshow("bgr",image_hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
	
	
	
	
	
	
	
	
	
	
	
	
	
	
# -------------------------------------------
# FINAL THRESHOLD:

# [ 76,  26, 198]
      # TO
# [156, 106, 278]