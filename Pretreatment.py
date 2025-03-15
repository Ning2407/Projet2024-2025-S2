import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def is_black_region(img, x1, y1, x2, y2, offset=10, threshold=0.8, min_area=500):
    dx = x2 - x1
    dy = y2 - y1
    norm_x, norm_y = -dy, dx
    length = np.sqrt(norm_x**2 + norm_y**2)
    norm_x, norm_y = norm_x / length, norm_y / length
    side1_x1, side1_y1 = int(x1 + norm_x * offset), int(y1 + norm_y * offset)
    side1_x2, side1_y2 = int(x2 + norm_x * offset), int(y2 + norm_y * offset)
    side2_x1, side2_y1 = int(x1 - norm_x * offset), int(y1 - norm_y * offset)
    side2_x2, side2_y2 = int(x2 - norm_x * offset), int(y2 - norm_y * offset)
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.line(mask, (side1_x1, side1_y1), (side1_x2, side1_y2), 255, 5)
    cv2.line(mask, (side2_x1, side2_y1), (side2_x2, side2_y2), 255, 5)
    region1 = img[mask == 255]
    region2 = img[mask == 255]
    area1 = len(region1)
    area2 = len(region2)
    black_ratio1 = np.sum(region1 == 0) / area1 if area1 > 0 else 0
    black_ratio2 = np.sum(region2 == 0) / area2 if area2 > 0 else 0
    return black_ratio1 > threshold and black_ratio2 > threshold and area1 > min_area and area2 > min_area

def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT)
    return rotated

def rotate_to_vertical(image_path):
    img=cv2.imread(image_path)
    if img is None:
        print("Image not loaded")
        return  
    img_region = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    start_gap = 1
    end_gap = 7
    for gap in range(start_gap, end_gap + 1):
        lines = cv2.HoughLinesP(
        edged, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=30, 
        maxLineGap=gap
    )
        valid_lines=[]
        for line in lines:
            x1,y1,x2,y2=line[0]
            if is_black_region(edged,x1,y1,x2,y2,50,0.95,700):
                valid_lines.append(line)
                cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2) 
        angles = []
        if lines is not None:
            for valid_line in valid_lines:
                x1, y1, x2, y2 = valid_line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angles.append(angle)
        print(angles)
        angel=np.asarray(angles)
        angel=angel[(angel!= 90.0)&(angel!= -90.0)&(angel!= 0.0)]
        sorted_angels = np.sort(angel)
        max_count = 0
        best_subset = []
        left = 0
        for right in range(len(sorted_angels)):
            while sorted_angels[right] - sorted_angels[left] > 1.1:
                left += 1  
            current_count = right - left + 1
            if current_count > max_count:
                max_count = current_count
                best_subset = sorted_angels[left:right+1]
        if len(best_subset) > 0:
            used_gap = gap
            break
    rotating_angle = np.mean(best_subset)
    if(rotating_angle<0):
        rotated_img = rotate_image(img_region, 90+rotating_angle)
    else:
        rotated_img = rotate_image(img_region, 90+rotating_angle)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    return img,contours,edged,rotated_img,angles,img_region
origin, contours, edged, rotated, angels, img_reg = rotate_to_vertical(r"D:\Code\sci-calcu\Challenge data\input_train\23b24c5d505f5d1cdf1d831a814f5cf68366fc29b96f7f2038be65a5f3d65944.png")
height, width, channels = rotated.shape
print(f"图像尺寸: {width}x{height}, 通道数: {channels}")

gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
contour_image = rotated.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)  # 绿色轮廓
plt.figure(figsize=(10, 5))
plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Detected Contours")
plt.show()

# 630*540
"""
def corp_720(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped_array = image_rgb[50:750, 275:430]
    return cropped_array
def corp_540(img):
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped_array = image_rgb[0:630, 200:330]
    return cropped_array

if(width==540):
    corped=corp_540(rotated)
elif(width==720):
    corped=corp_720(rotated)
"""
"""


if edged is not None:

    plt.subplot(1, 3, 1)
    plt.imshow(corped, cmap="gray")  # `cmap="gray"` 让灰度图显示正确
    plt.title("corped")
    plt.axis("off")  # 关闭坐标轴


    plt.subplot(1, 3, 2)
    plt.imshow(edged, cmap="gray")  # `cmap="gray"` 让灰度图显示正确
    plt.title("Edge Detection Result")
    plt.axis("off")  # 关闭坐标轴


    plt.subplot(1, 3, 3)
    plt.imshow(rotated, cmap="gray")  # `cmap="gray"` 让灰度图显示正确
    plt.title("region")
    plt.axis("off")  # 关闭坐标轴
    plt.show()"
"""