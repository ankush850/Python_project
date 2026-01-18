# Import necessary modules
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2 as cv
import os

def main():
    # Make img folder to run this code on image which u want to test 
    # Replace 'img/road_1.jpeg' with the path to your road image
    image_path = 'img/road_1.jpeg'
    try:
        image = cv.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image not found at {image_path}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Get image dimensions
    height = image.shape[0]
    width = image.shape[1]
    print(f"Image dimensions: {width} x {height}")
    
    # Create a copy of the image for processing
    img_copy = np.copy(image)
    
    # Step 2: Define Region of Interest (ROI) function
    def region_of_interest(img, vertices):
        """
        Applies a mask to the image to focus on a specific region of interest.
        
        Parameters:
        - img: Input image (numpy array)
        - vertices: List of vertices defining the polygon for the ROI
        
        Returns:
        - masked_image: Image with only the ROI visible
        """
        mask = np.zeros_like(img)
        # If the image has multiple channels, use the first channel for masking
        if len(img.shape) > 2:
            channel_count = img.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
        
        cv.fillPoly(mask, vertices, ignore_mask_color)
        masked_image = cv.bitwise_and(img, mask)
        return masked_image
    
    # Define vertices for the region of interest (triangle focusing on the road ahead)
    region_of_interest_vertices = [
        (0, height),  # Bottom-left
        (width / 2, height / 2),  # Top-center
        (width, height),  # Bottom-right
    ]
    
    # Step 3: Convert to grayscale and apply Canny edge detection
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    canny = cv.Canny(gray, 100, 200)  # Thresholds: 100 (min), 200 (max)
    
    # Apply ROI mask to the Canny edges
    cropped_image = region_of_interest(canny, np.array([region_of_interest_vertices], np.int32))
    
    # Display the cropped Canny image
    plt.figure(figsize=(10, 6))
    plt.imshow(cropped_image, cmap='gray')
    plt.title('Cropped Canny Edge Detection')
    plt.axis('off')
    plt.show()
    
    # Step 4: Laplacian Edge Detection
    # Read the image in grayscale (reusing gray from above for efficiency)
    img_gray = gray  # Already computed
    
    # Remove noise using Gaussian blur
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    
    # Apply Laplacian operator for edge detection
    laplacian = cv.Laplacian(img_blur, cv.CV_64F)
    laplacian = cv.convertScaleAbs(laplacian)  # Convert to uint8 for display
    
    # Display original and Laplacian images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_blur, cmap='gray')
    plt.title('Blurred Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(laplacian, cmap='gray')
    plt.title('Laplacian Edge Detection')
    plt.axis('off')
    plt.show()
    
    # Step 5: Sobel Edge Detection (using OpenCV's built-in Sobel for better accuracy)
    # Apply Sobel X and Y
    sobel_x = cv.Sobel(img_gray, cv.CV_64F, 1, 0, ksize=3)
    sobel_y = cv.Sobel(img_gray, cv.CV_64F, 0, 1, ksize=3)
    
    # Convert to absolute values and uint8
    sobel_x = cv.convertScaleAbs(sobel_x)
    sobel_y = cv.convertScaleAbs(sobel_y)
    
    # Display grayscale image and Sobel results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(sobel_x, cmap='gray')
    plt.title('Sobel X (Horizontal Edges)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(sobel_y, cmap='gray')
    plt.title('Sobel Y (Vertical Edges)')
    plt.axis('off')
    plt.show()
    
    # Step 6: Prewitt Edge Detection (additional method for comparison)
    # Prewitt kernels
    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    
    # Apply Prewitt filters
    prewitt_x_img = cv.filter2D(img_gray, -1, prewitt_x)
    prewitt_y_img = cv.filter2D(img_gray, -1, prewitt_y)
    
    # Display Prewitt results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(prewitt_x_img, cmap='gray')
    plt.title('Prewitt X')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(prewitt_y_img, cmap='gray')
    plt.title('Prewitt Y')
    plt.axis('off')
    plt.show()
    
    # Step 7: Roberts Edge Detection (another classic method)
    # Roberts kernels
    roberts_x = np.array([[1, 0], [0, -1]])
    roberts_y = np.array([[0, 1], [-1, 0]])
    
    # Apply Roberts filters
    roberts_x_img = cv.filter2D(img_gray, -1, roberts_x)
    roberts_y_img = cv.filter2D(img_gray, -1, roberts_y)
    
    # Combine Roberts edges
    roberts_combined = cv.addWeighted(roberts_x_img, 0.5, roberts_y_img, 0.5, 0)
    
    # Display Roberts results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(roberts_combined, cmap='gray')
    plt.title('Roberts Edge Detection')
    plt.axis('off')
    plt.show()
    
    # Step 8: Canny Edge Detection with different parameters (improved thresholds)
    # Apply Gaussian blur first
    img_blur_canny = cv.GaussianBlur(img_gray, (5, 5), 0)
    
    # Canny edge detection with adjusted thresholds (lower for more edges)
    edges = cv.Canny(img_blur_canny, 50, 150)
    
    # Display original blurred image and Canny edges
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_blur_canny, cmap='gray')
    plt.title('Blurred Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection (Adjusted Thresholds)')
    plt.axis('off')
    plt.show()
    
    # Step 9: Additional Processing - Combine edges from different methods
    # Combine Sobel X and Y for gradient magnitude
    sobel_combined = cv.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)
    
    # Combine with Canny for a hybrid edge image
    combined_edges = cv.bitwise_or(sobel_combined, edges)
    
    # Display combined edges
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Combined Sobel Edges')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(combined_edges, cmap='gray')
    plt.title('Hybrid Edges (Sobel + Canny)')
    plt.axis('off')
    plt.show()
    
    # Step 10: Color-based Lane Detection using HSV color space
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Define range for white lanes
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 30, 255])
    mask_white = cv.inRange(hsv, lower_white, upper_white)
    
    # Define range for yellow lanes
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv.inRange(hsv, lower_yellow, upper_yellow)
    
    # Combine masks
    mask_lane = cv.bitwise_or(mask_white, mask_yellow)
    
    # Apply mask to original image
    lane_image = cv.bitwise_and(image, image, mask=mask_lane)
    
    # Display color-based detection
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv.cvtColor(lane_image, cv.COLOR_BGR2RGB))
    plt.title('Color-based Lane Detection')
    plt.axis('off')
    plt.show()
    
    # Step 11: Combine Color and Edge Detection
    # Dilate the lane mask slightly
    kernel = np.ones((5,5), np.uint8)
    mask_lane_dilated = cv.dilate(mask_lane, kernel, iterations=1)
    
    # Combine with edges
    final_mask = cv.bitwise_and(combined_edges, mask_lane_dilated)
    
    # Display combined result
    plt.figure(figsize=(8, 6))
    plt.imshow(final_mask, cmap='gray')
    plt.title('Combined Color and Edge Detection')
    plt.axis('off')
    plt.show()
    
    # Step 12: Perspective Transformation for Bird's Eye View
    # Define source points (trapezoid on road)
    src = np.float32([
        [width * 0.1, height * 0.9],  # Bottom-left
        [width * 0.9, height * 0.9],  # Bottom-right
        [width * 0.6, height * 0.6],  # Top-right
        [width * 0.4, height * 0.6]   # Top-left
    ])
    
    # Define destination points (rectangle)
    dst = np.float32([
        [0, height],
        [width, height],
        [width, 0],
        [0, 0]
    ])
    
    # Get perspective transform matrix
    M = cv.getPerspectiveTransform(src, dst)
    
    # Apply warp
    warped = cv.warpPerspective(final_mask, M, (width, height))
    
    # Display warped image
    plt.figure(figsize=(10, 6))
    plt.imshow(warped, cmap='gray')
    plt.title('Bird\'s Eye View of Lanes')
    plt.axis('off')
    plt.show()
    
    # Step 13: Hough Line Transform for lane detection (improved with averaging)
    # Use the warped image for Hough transform
    lines = cv.HoughLinesP(warped, 1, np.pi/180, 50, minLineLength=100, maxLineGap=50)
    
    # Function to average lines
    def average_lines(lines):
        left_lines = []
        right_lines = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1) if x2 != x1 else 0
                if slope < 0:  # Left lane (negative slope)
                    left_lines.append((slope, y1 - slope * x1))
                elif slope > 0:  # Right lane (positive slope)
                    right_lines.append((slope, y1 - slope * x1))
        
        # Average left lines
        if left_lines:
            left_avg = np.average(left_lines, axis=0)
            left_line = make_line_points(left_avg, height)
        else:
            left_line = None
        
        # Average right lines
        if right_lines:
            right_avg = np.average(right_lines, axis=0)
            right_line = make_line_points(right_avg, height)
        else:
            right_line = None
        
        return left_line, right_line
    
    def make_line_points(avg, height):
        slope, intercept = avg
        y1 = height
        y2 = int(height * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return [x1, y1, x2, y2]
    
    # Get averaged lines
    left_line, right_line = average_lines(lines)
    
    # Draw lines on the original image
    line_image = np.copy(image)
    if left_line:
        cv.line(line_image, (left_line[0], left_line[1]), (left_line[2], left_line[3]), (0, 255, 0), 10)
    if right_line:
        cv.line(line_image, (right_line[0], right_line[1]), (right_line[2], right_line[3]), (0, 255, 0), 10)
    
    # Display image with detected lanes
    plt.figure(figsize=(10, 6))
    plt.imshow(cv.cvtColor(line_image, cv.COLOR_BGR2RGB))
    plt.title('Averaged Lane Detection')
    plt.axis('off')
    plt.show()
    
    # Step 14: Polynomial Fitting for Curved Lanes (advanced)
    # Extract points from lines for polynomial fit
    if lines is not None:
        lane_points = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            lane_points.append([x1, y1])
            lane_points.append([x2, y2])
        
        if lane_points:
            lane_points = np.array(lane_points)
            # Fit polynomial (degree 2 for curves)
            left_fit = np.polyfit(lane_points[lane_points[:, 0] < width/2, 1], lane_points[lane_points[:, 0] < width/2, 0], 2)
            right_fit = np.polyfit(lane_points[lane_points[:, 0] >= width/2, 1], lane_points[lane_points[:, 0] >= width/2, 0], 2)
            
            # Generate points for plotting
            ploty = np.linspace(0, height-1, height)
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            
            # Draw polynomial lines
            poly_image = np.copy(image)
            for i in range(len(ploty)-1):
                cv.line(poly_image, (int(left_fitx[i]), int(ploty[i])), (int(left_fitx[i+1]), int(ploty[i+1])), (255, 0, 0), 5)
                cv.line(poly_image, (int(right_fitx[i]), int(ploty[i])), (int(right_fitx[i+1]), int(ploty[i+1])), (255, 0, 0), 5)
            
            # Display polynomial lanes
            plt.figure(figsize=(10, 6))
            plt.imshow(cv.cvtColor(poly_image, cv.COLOR_BGR2RGB))
            plt.title('Polynomial Lane Fitting')
            plt.axis('off')
            plt.show()
    
    print("Road detection processing complete!")

if __name__ == "__main__":
    main()
