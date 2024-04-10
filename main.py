import cv2
import numpy as np
from sklearn.cluster import KMeans

def cluster_lines_based_on_position_and_angle(lines):
    """
    Cluster lines based on their position and angle using KMeans algorithm.
    
    :param lines: List of lines where each line is represented as [x1, y1, x2, y2].
    :return: A list of clusters, where each cluster contains lines belonging to that cluster.
    """
    # Initialize KMeans with 2 clusters.
    kmeans = KMeans(n_clusters=2)
    
    # Prepare features for clustering: positions (start and end points) and geometrical properties (length, angle) of lines.
    features = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        angle = np.arctan2(y2 - y1, x2 - x1)
        features.append([x1, y1, x2, y2, length, angle])
    
    # Perform clustering.
    kmeans.fit(features)
    
    # Organize lines into clusters based on the labels assigned by KMeans.
    clusters = [[] for _ in range(kmeans.n_clusters)]
    [clusters[label].append(line) for label, line in zip(kmeans.labels_, lines)]
    return clusters

def calculate_average_line_for_cluster(lines):
    """
    Calculate the average line for a cluster of lines.
    
    :param lines: List of lines belonging to a cluster.
    :return: Coordinates of the middle line [x1, y1, x2, y2].
    """
    # Calculate the average start and end points of the lines in the cluster.
    starts = np.mean([line[0][:2] for line in lines], axis=0)
    ends = np.mean([line[0][2:] for line in lines], axis=0)
    
    # Determine the middle point between the average start and end points.
    middle_point = ((starts + ends) / 2).astype(int)

    # Calculate angle and set a fixed length for the resulting line.
    angle = np.arctan2(ends[1] - starts[1], ends[0] - starts[0])
    fixed_length = 320

    # Determine the new line's start and end points based on the fixed length.
    delta_x = int(fixed_length * np.cos(angle))
    delta_y = int(fixed_length * np.sin(angle))

    x1, y1 = middle_point[0] - delta_x, middle_point[1] - delta_y
    x2, y2 = middle_point[0] + delta_x, middle_point[1] + delta_y

    return [x1, y1, x2, y2]

def find_lines_intersection(first_line, second_line):
    """
    Calculate the intersection point of two lines.
    
    :param first_line: First line represented by start and end points [x1, y1, x2, y2].
    :param second_line: Second line represented by start and end points [x3, y3, x4, y4].
    :return: The intersection point (x, y) or None if the lines are parallel.
    """
    (x1, y1, x2, y2), (x3, y3, x4, y4) = first_line, second_line

    # Initialize slopes to infinity.
    m1 = float('inf')
    m2 = float('inf')

    # Calculate slopes of the lines if possible.
    if x2 - x1 != 0:
        m1 = (y2 - y1) / (x2 - x1)
    if x4 - x3 != 0:
        m2 = (y4 - y3) / (x4 - x3)

    # Return None if lines are parallel (slopes are equal).
    if m1 == m2:
        return None

    # Calculate the intersection point.
    x = ((m1 * x1 - y1) - (m2 * x3 - y3)) / (m1 - m2)
    y = m1 * (x - x1) + y1

    return int(x), int(y)

def extract_significant_lines_and_intersection(image):
    """
    Extract significant lines from the image and find their intersection point.
    
    :param image: The input image.
    :return: A tuple of significant lines and their intersection point.
    """
    # Detect lines using Hough Transform.
    lines = cv2.HoughLinesP(image, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
    
    # Cluster lines to find significant lines.
    clusters = cluster_lines_based_on_position_and_angle(lines)
    
    # Calculate the average line for each cluster.
    significant_lines = [calculate_average_line_for_cluster(cluster) for cluster in clusters]
    
    # Find the intersection point of the two significant lines.
    intersection_point = find_lines_intersection(significant_lines[0], significant_lines[1])
    
    return significant_lines, intersection_point

def preprocess_image_for_line_extraction(image_path):
    """
    Preprocess the image to enhance line features.
    
    :param image_path: Path to the image file.
    :return: Preprocessed image.
    """
    # Read the image in grayscale.
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply binary thresholding.
    _, mask = cv2.threshold(image, 210, 255, cv2.THRESH_BINARY)
    
    # Use the mask to isolate the lines.
    image = cv2.bitwise_and(image, image, mask=mask)
    
    # Apply Gaussian Blur multiple times to smooth the image.
    for _ in range(7):
        image = cv2.GaussianBlur(image, (5,5), 0)
    
    # Use Canny edge detection.
    image = cv2.Canny(image, 240, 250)
    
    return image

def visualize_results_on_image(image, lines, intersection_point):
    """
    Draw lines and intersection point on the image.
    
    :param image: The original image.
    :param lines: List of lines to be drawn.
    :param intersection_point: The intersection point to be marked.
    """
    if not lines:
        return
    
    # Draw each line in green with a thickness of 5.
    for line in lines:
        cv2.line(image, (line[0], line[1]), (line[2], line[3]), (0, 255, 0), 5)

    # Draw the intersection point as a red circle.
    if intersection_point:
        cv2.circle(image, intersection_point, 5, (0, 0, 255), -1)


# Main process
path_to_file = "src/image.jpg"

# Load the original image
original_image = cv2.imread(path_to_file)

# Preprocess the image for line extraction
processed_image = preprocess_image_for_line_extraction(path_to_file)

# Extract significant lines and their intersection point
result_lines, intersection_point = extract_significant_lines_and_intersection(processed_image)

# Draw results on the original image
visualize_results_on_image(original_image, result_lines, intersection_point)

out_path = "src/out.jpg"
# Save the result
cv2.imwrite(out_path, original_image)

print("Saved the new image into: ", out_path)
