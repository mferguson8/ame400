import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend to non-interactive
import matplotlib.pyplot as plt
import sys
import os

def save_plot(filename):
    """Helper function to save plots instead of displaying them"""
    plt.savefig(filename)
    plt.close()

########## Section 2.1 - Load and display images with OpenCV #######################
# 2.a Read in the image and display with Matplotlib (colors will look inverted)
img_bgr = cv2.imread('elephant.jpeg')
if img_bgr is None:
    print("Error: Could not load image 'elephant.jpeg'")
    print("Please ensure the image file exists in:", os.getcwd())
    sys.exit(1)

plt.figure()
plt.imshow(img_bgr)
plt.title('Loaded with cv2 (BGR interpreted as RGB)')
plt.axis('off')
save_plot('display_bgr.png')
# Write out with OpenCV
cv2.imwrite('elephant_opencv.png', img_bgr)

# 2.b Convert BGR to RGB for Matplotlib and display
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img_rgb)
plt.title('Converted to RGB for Matplotlib')
plt.axis('off')
save_plot('display_rgb.png')
# Write out converted image
cv2.imwrite('elephant_matplotlib.png', img_rgb)

# 2.c Read/convert to grayscale, display, and write out
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
plt.figure()
plt.imshow(img_gray, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
save_plot('display_gray.png')
# Write grayscale image
cv2.imwrite('elephant_gray.png', img_gray)

########## Section 3 - Cropping ##############################################
# Crop out the small elephant; adjust coordinates as needed
# Here we choose row 200:900, col 50:650 based on inspection
y1, y2 = 200, 900
x1, x2 = 50, 650
baby_rgb = img_rgb[y1:y2, x1:x2]
plt.figure()
plt.imshow(baby_rgb)
plt.title('Cropped Baby Elephant')
plt.axis('off')
save_plot('display_crop.png')
# Convert back to BGR for writing
baby_bgr = cv2.cvtColor(baby_rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite('babyelephant.png', baby_bgr)

########## Section 4 - Pixel-wise Arithmetic Operations ####################
# 4.a Prepare an RGB copy
arith_rgb = img_rgb.astype(np.int16)  # Convert to int16 to handle overflow

# 4.b Add 256 using NumPy
arith_np = arith_rgb + 256
print('After numpy add, dtype:', arith_np.dtype)
print('  min/max before cast:', arith_np.min(), arith_np.max())
arith_cast = np.clip(arith_np, 0, 255).astype(np.uint8)  # Clip values and convert back to uint8
print('After uint8 cast, dtype:', arith_cast.dtype)
plt.figure()
plt.imshow(arith_cast)
plt.title('NumPy add +256 then uint8 cast')
plt.axis('off')
save_plot('display_numpy_add.png')

# 4.c Add 256 using OpenCV (saturating arithmetic)
# Work on original BGR for clarity
b, g, r = cv2.split(img_bgr)
b_add = cv2.add(b, 256)
g_add = cv2.add(g, 256)
r_add = cv2.add(r, 256)
merged_bgr = cv2.merge([b_add, g_add, r_add])
merged_rgb = cv2.cvtColor(merged_bgr, cv2.COLOR_BGR2RGB)
plt.figure()
plt.imshow(merged_rgb)
plt.title('OpenCV add 256 (saturated)')
plt.axis('off')
plt.show()

########## Section 5 - Resizing Images ######################################
# 5.a Read and convert to RGB (already have img_rgb)

# 5.b Downsample by 10x
h, w = img_rgb.shape[:2]
down = cv2.resize(img_rgb, (w//10, h//10), interpolation=cv2.INTER_AREA)
plt.figure()
plt.imshow(down)
plt.title('10x Downsampled')
plt.axis('off')
plt.show()
cv2.imwrite('elephant_10xdown.png', cv2.cvtColor(down, cv2.COLOR_RGB2BGR))

# 5.c Upsample back using Nearest Neighbor and Bicubic
up_nn = cv2.resize(down, (w, h), interpolation=cv2.INTER_NEAREST)
up_bc = cv2.resize(down, (w, h), interpolation=cv2.INTER_CUBIC)
plt.figure(); plt.imshow(up_nn); plt.title('Upsampled Nearest Neighbor'); plt.axis('off'); plt.show()
plt.figure(); plt.imshow(up_bc); plt.title('Upsampled Bicubic'); plt.axis('off'); plt.show()
cv2.imwrite('elephant_10xup_nearest.png', cv2.cvtColor(up_nn, cv2.COLOR_RGB2BGR))
cv2.imwrite('elephant_10xup_bicubic.png', cv2.cvtColor(up_bc, cv2.COLOR_RGB2BGR))

# 5.d Compute absolute difference and error sums
diff_nn = cv2.absdiff(img_rgb, up_nn)
diff_bc = cv2.absdiff(img_rgb, up_bc)
cv2.imwrite('diff_nearest.png', cv2.cvtColor(diff_nn, cv2.COLOR_RGB2BGR))
cv2.imwrite('diff_bicubic.png', cv2.cvtColor(diff_bc, cv2.COLOR_RGB2BGR))
error_nn = np.sum(diff_nn)
error_bc = np.sum(diff_bc)
print(f'Error sum (Nearest): {error_nn}')
print(f'Error sum (Bicubic): {error_bc}')
if error_nn < error_bc:
    print('Nearest neighbor caused less error.')
else:
    print('Bicubic caused less error.')
