# Import necessary modules!
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2 as cv

def main():
    """
    Main function to execute the road detection program.
    This includes loading the image, applying various edge detection techniques,
    and displaying the results.
    """
    
    # Step 1: Load the image
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
    # Read the image in grayscale
    img_gray = cv.imread(image_path, 0)
    
    # Remove noise using Gaussian blur
    img_blur = cv.GaussianBlur(img_gray, (3, 3), 0)
    
    # Apply Laplacian operator for edge detection
    laplacian = cv.Laplacian(img_blur, cv.CV_64F)
    
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
    
    # Step 5: Sobel Edge Detection
    # Define Sobel kernels
    # Sobel Y kernel (detects vertical edges)
    sobel_y = np.array([[ -1, -2, -1], 
                        [  0,  0,  0], 
                        [  1,  2,  1]])
    
    # Sobel X kernel (detects horizontal edges)
    sobel_x = np.array([[ -1,  0,  1],
                        [ -2,  0,  2],
                        [ -1,  0,  1]])
    
    # Apply Sobel filters using cv.filter2D
    filtered_image_y = cv.filter2D(img_gray, -1, sobel_y)
    filtered_image_x = cv.filter2D(img_gray, -1, sobel_x)
    
    # Display grayscale image and Sobel results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_gray, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(filtered_image_x, cmap='gray')
    plt.title('Sobel X (Horizontal Edges)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(filtered_image_y, cmap='gray')
    plt.title('Sobel Y (Vertical Edges)')
    plt.axis('off')
    plt.show()
    
    # Step 6: Canny Edge Detection with different parameters
    # Apply Gaussian blur first
    img_blur_canny = cv.GaussianBlur(img_gray, (5, 5), 0)
    
    # Canny edge detection with adjusted thresholds
    edges = cv.Canny(img_blur_canny, 250, 250)
    
    # Display original blurred image and Canny edges
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(img_blur_canny, cmap='gray')
    plt.title('Blurred Grayscale Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection')
    plt.axis('off')
    plt.show()
    
    # Step 7: Additional Processing - Combine edges
    # Combine Sobel X and Y for gradient magnitude
    sobel_combined = cv.addWeighted(filtered_image_x, 0.5, filtered_image_y, 0.5, 0)
    
    # Display combined Sobel
    plt.figure(figsize=(8, 6))
    plt.imshow(sobel_combined, cmap='gray')
    plt.title('Combined Sobel Edges')
    plt.axis('off')
    plt.show()
    
    # Step 8: Hough Line Transform for lane detection (basic example)
    # Use the Canny edges for Hough transform
    lines = cv.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    # Draw lines on the original image
    line_image = np.copy(image)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
    
    # Display image with detected lines
    plt.figure(figsize=(10, 6))
    plt.imshow(cv.cvtColor(line_image, cv.COLOR_BGR2RGB))
    plt.title('Hough Line Transform for Lane Detection')
    plt.axis('off')
    plt.show()
    
    print("Road detection processing complete!")

if __name__ == "__main__":
    main()
