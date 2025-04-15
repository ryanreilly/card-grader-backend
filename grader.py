import cv2
import numpy as np
from scipy.stats import entropy

def preprocess_image(image):
    """Preprocess the image to improve analysis accuracy, with robust card detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v_eq = cv2.equalizeHist(v)
    hsv_eq = cv2.merge((h, s, v_eq))
    image_eq = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)
    blurred = cv2.GaussianBlur(image_eq, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return blurred
    card_contour = None
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > 10000:
            card_contour = contour
            break
    if card_contour is None:
        return blurred
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [card_contour], -1, 255, -1)
    masked_image = cv2.bitwise_and(blurred, blurred, mask=mask)
    return masked_image

def correct_orientation(image):
    """Detect and correct the card's orientation to ensure it is upright."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    if lines is None or len(lines) == 0:
        return image
    
    rho, theta = lines[0][0]
    angle = (theta * 180 / np.pi) - 90
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def calculate_centering_ratio(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0, (0, 0, 0, 0)
    card_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(card_contour)
    card_img = image[y:y+h, x:x+w]
    card_gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
    content_edges = cv2.Canny(card_gray, 50, 150)
    rows_with_edges = [row for row in content_edges if row.any()]
    if not rows_with_edges:
        return 0, (0, 0, 0, 0)
    left_edge = min([row.nonzero()[0].min() for row in rows_with_edges])
    right_edge = max([row.nonzero()[0].max() for row in rows_with_edges])
    cols_with_edges = [col for col in range(content_edges.shape[1]) if content_edges[:, col].any()]
    if not cols_with_edges:
        return 0, (0, 0, 0, 0)
    top_edge = min([content_edges[:, col].nonzero()[0].min() for col in cols_with_edges])
    bottom_edge = max([content_edges[:, col].nonzero()[0].max() for col in cols_with_edges])
    left_border = left_edge
    right_border = w - right_edge - 1
    top_border = top_edge
    bottom_border = h - bottom_edge - 1
    if min(left_border, right_border, top_border, bottom_border) <= 0:
        return 0, (0, 0, 0, 0)
    horizontal_ratio = min(left_border, right_border) / max(left_border, right_border)
    vertical_ratio = min(top_border, bottom_border) / max(top_border, bottom_border)
    centering_ratio = min(horizontal_ratio, vertical_ratio)
    return centering_ratio, (left_border, right_border, top_border, bottom_border)

def assess_corners(image):
    """Assess corner sharpness using average gradient magnitude at the four corners."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    card_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, epsilon, True)
    if len(approx) == 4:
        corner_points = approx.reshape(4, 2)
    else:
        x, y, w, h = cv2.boundingRect(card_contour)
        corner_points = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    corner_gradients = []
    for pt in corner_points:
        x, y = pt
        if 0 <= y < gradient_magnitude.shape[0] and 0 <= x < gradient_magnitude.shape[1]:
            corner_gradients.append(gradient_magnitude[y, x])
        else:
            corner_gradients.append(0)
    return np.mean(corner_gradients)

def assess_edges(image):
    """Assess edge straightness by measuring deviation from fitted lines."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0
    card_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(card_contour, True)
    approx = cv2.approxPolyDP(card_contour, epsilon, True)
    if len(approx) == 4:
        points = approx.reshape(4, 2)
    else:
        x, y, w, h = cv2.boundingRect(card_contour)
        points = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    deviations = []
    for i in range(4):
        p1 = points[i]
        p2 = points[(i+1)%4]
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        a = vy
        b = -vx
        c = vx * p1[1] - vy * p1[0]
        norm = np.sqrt(a**2 + b**2)
        if norm == 0:
            deviations.append(0)
            continue
        a /= norm
        b /= norm
        c /= norm
        contour_points = card_contour.reshape(-1, 2).astype(float)
        distances = np.abs(a * contour_points[:,0] + b * contour_points[:,1] + c)
        close_points = distances < 20
        if np.any(close_points):
            deviations.append(np.mean(distances[close_points]))
        else:
            deviations.append(0)
    average_deviation = np.mean(deviations)
    max_deviation = 50
    normalized_deviation = min(average_deviation, max_deviation) / max_deviation
    edge_score = 1 - normalized_deviation
    return edge_score

def assess_surface(image, color_threshold=60, edge_low=200, edge_high=400, line_threshold=300):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    
    total_pixels = image.shape[0] * image.shape[1]
    defect_map = np.zeros_like(gray, dtype=np.uint8)
    
    edges = cv2.Canny(gray, edge_low, edge_high)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)
    scratch_defects = np.zeros_like(gray, dtype=np.uint8)
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(scratch_defects, (x1, y1), (x2, y2), 255, 2)
    contours, _ = cv2.findContours(scratch_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1500:
            cv2.drawContours(defect_map, [contour], -1, 255, -1)
    
    h_var = cv2.GaussianBlur(h, (5, 5), 0)
    s_var = cv2.GaussianBlur(s, (5, 5), 0)
    h_var = cv2.convertScaleAbs(cv2.Sobel(h_var, cv2.CV_64F, 1, 1, ksize=3))
    s_var = cv2.convertScaleAbs(cv2.Sobel(s_var, cv2.CV_64F, 1, 1, ksize=3))
    color_defects = cv2.bitwise_or(h_var, s_var)
    _, color_defects = cv2.threshold(color_defects, color_threshold, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    color_defects = cv2.morphologyEx(color_defects, cv2.MORPH_OPEN, kernel, iterations=2)
    defect_map = cv2.bitwise_or(defect_map, color_defects)
    
    edges_faint = cv2.Canny(gray, edge_low // 2, edge_high // 2)
    lines_faint = cv2.HoughLines(edges_faint, 1, np.pi / 180, line_threshold // 2)
    print_defects = np.zeros_like(gray, dtype=np.uint8)
    if lines_faint is not None:
        for line in lines_faint:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(print_defects, (x1, y1), (x2, y2), 255, 1)
    
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    v_std = cv2.convertScaleAbs(cv2.Sobel(v_blur, cv2.CV_64F, 1, 1, ksize=3))
    _, factory_defects = cv2.threshold(v_std, 50, 255, cv2.THRESH_BINARY)
    factory_defects = cv2.morphologyEx(factory_defects, cv2.MORPH_OPEN, kernel, iterations=2)
    
    other_defects = cv2.bitwise_or(print_defects, factory_defects)
    defect_map = cv2.bitwise_or(defect_map, other_defects)
    
    scratch_score = cv2.countNonZero(scratch_defects) / total_pixels * 0.3
    color_score = cv2.countNonZero(color_defects) / total_pixels * 1.0
    print_score = cv2.countNonZero(print_defects) / total_pixels * 0.1
    factory_score = cv2.countNonZero(factory_defects) / total_pixels * 0.5
    
    surface_score = scratch_score + color_score + print_score + factory_score
    
    print(f"Scratch Score: {scratch_score:.4f}")
    print(f"Color Score: {color_score:.4f}")
    print(f"Print Score: {print_score:.4f}")
    print(f"Factory Score: {factory_score:.4f}")
    print(f"Total Surface Score: {surface_score:.4f}")
    
    return surface_score

def grade_centering(centering_ratio, centering_thresholds):
    """Grade the centering based on the centering ratio."""
    for grade in range(10, 0, -1):
        if centering_ratio >= centering_thresholds[grade]:
            return grade
    return 1

def grade_corners(corner_score, corner_thresholds):
    """Grade the corners based on the corner score."""
    for grade in range(10, 0, -1):
        if corner_score >= corner_thresholds[grade]:
            return grade
    return 1

def grade_edges(edge_score, edge_thresholds):
    """Grade the edges based on the edge score."""
    for grade in range(10, 0, -1):
        if edge_score >= edge_thresholds[grade]:
            return grade
    return 1

def grade_surface(surface_score, surface_thresholds):
    """Grade the surface based on the surface score."""
    for grade in range(10, 0, -1):
        if surface_score <= surface_thresholds[grade]:
            return grade
    return 1

def grade_card(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image could not be loaded.")
    corrected_image = correct_orientation(image)
    preprocessed_image = preprocess_image(corrected_image)
    
    centering_result = calculate_centering_ratio(preprocessed_image)
    centering_ratio, borders = centering_result
    corner_score = assess_corners(preprocessed_image)
    edge_score = assess_edges(preprocessed_image)
    surface_score = assess_surface(preprocessed_image)
    
    centering_thresholds = {
        10: 0.818, 9: 0.667, 8: 0.538, 7: 0.429, 6: 0.250,
        5: 0.176, 4: 0.176, 3: 0.111, 2: 0.111, 1: 0.0
    }
    corner_thresholds = {
        10: 100, 9: 95, 8: 90, 7: 85, 6: 75,
        5: 65, 4: 55, 3: 45, 2: 35, 1: 0
    }
    edge_thresholds = {
        10: 0.90, 9: 0.85, 8: 0.80, 7: 0.70, 6: 0.60,
        5: 0.50, 4: 0.40, 3: 0.30, 2: 0.20, 1: 0.0
    }
    surface_thresholds = {
        10: 0.02, 9: 0.05, 8: 0.10, 7: 0.15, 6: 0.20,
        5: 0.30, 4: 0.40, 3: 0.50, 2: 0.60, 1: 0.80
    }
    
    centering_grade = grade_centering(centering_ratio, centering_thresholds)
    corners_grade = grade_corners(corner_score, corner_thresholds)
    edges_grade = grade_edges(edge_score, edge_thresholds)
    print(f"Surface Score Before Grading: {surface_score:.4f}")
    surface_grade = grade_surface(surface_score, surface_thresholds)
    print(f"Surface Grade: {surface_grade}")
    
    overall_grade = min(centering_grade, corners_grade, edges_grade, surface_grade)
    
    return {
        "centering": centering_grade,
        "corners": corners_grade,
        "edges": edges_grade,
        "surface": surface_grade,
        "overall": overall_grade
    }

if __name__ == "__main__":
    image_path = "IMG_4116.jpg"
    grade_names = {
        10: "Gem Mint 10 (GM-MT)", 9: "Mint 9 (MINT)", 8: "Near Mint-Mint 8 (NM-MT)",
        7: "Near Mint 7 (NM)", 6: "Excellent-Mint 6 (EX-MT)", 5: "Excellent 5 (EX)",
        4: "Very Good-Excellent 4 (VG-EX)", 3: "Very Good 3 (VG)", 2: "Good 2 (GOOD)",
        1: "Poor 1 (PR) or Fair 1.5 (FR)"
    }
    
    try:
        grades = grade_card(image_path)
        print("Individual Grades:")
        print(f"  Centering: {grades['centering']} ({grade_names[grades['centering']]})")
        print(f"  Corners: {grades['corners']} ({grade_names[grades['corners']]})")
        print(f"  Edges: {grades['edges']} ({grade_names[grades['edges']]})")
        print(f"  Surface: {grades['surface']} ({grade_names[grades['surface']]})")
        print(f"\nOverall Grade: {grades['overall']} ({grade_names[grades['overall']]})")
    except Exception as e:
        print(f"Error processing the image: {e}")
