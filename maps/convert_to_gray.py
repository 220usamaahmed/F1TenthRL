import cv2
import glob

for file in glob.glob("*/*.png"):
    print(file)

    image = cv2.imread(file, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(file, gray)
