import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
model = YOLO('best.pt')
image_path = '13.jpg'
image = cv2.imread(image_path)
results = model(image)
print(results)
annotated_image = results[0].plot()
cv2.imwrite('annotated_image.jpg', annotated_image)
cv2.imshow('Annotated Image', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
plt.show()
