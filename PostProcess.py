import cv2
import numpy as np
import imutils

# post-process a stitched panorama image by adding borders, cropping to content and saving the result
def post_process(stitched_img, output_processed_path):
    # add border and prepare image
    bordered_img = add_border(stitched_img)

    # create binary mask of content area
    content_mask = create_content_mask(bordered_img)

    # find optimal crop area
    min_rect_mask = find_minimum_content_rect(content_mask)

    # crop to final dimensions
    cropped_img = crop_to_mask(bordered_img, min_rect_mask)

    # finishing touches by sharpening the images
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    final_img = cv2.filter2D(cropped_img, -1, kernel)

    # save the final panorama result
    cv2.imwrite(output_processed_path, final_img)
    print(f"Processed panorama saved as {output_processed_path}")

# add a black border around the image
def add_border(img, border_size=10, border_color=(0, 0, 0)):

    return cv2.copyMakeBorder(
        img,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        border_color
    )

# threshold grayscale image so that non-black areas become white mask
def create_content_mask(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                # convert to single chanel
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)    # auto threshold
    return thresh

# find the minimum rectangle that contains all content
def find_minimum_content_rect(mask):

    # find initial bounding rectangle
    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    largest_contour = max(contours, key=cv2.contourArea)        # pick the biggest contour

    x, y, w, h = cv2.boundingRect(largest_contour)  # get its bounding rectangle

    # create mask for the bounding rectangle
    rect_mask = np.zeros(mask.shape, dtype="uint8")
    cv2.rectangle(rect_mask, (x, y), (x + w, y + h), 255, -1)

    # erode until it fits content perfectly
    min_rect = rect_mask.copy()
    sub = rect_mask.copy()
    while cv2.countNonZero(sub) > 0:
        min_rect = cv2.erode(min_rect, None)    # shrink rectangle by 1px
        sub = cv2.subtract(min_rect, mask)             # difference from content mask

    return min_rect

# crop image to the given mask's bounding rectangle
def crop_to_mask(img, mask):

    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if not contours:
        return img  # return original if no contours found

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    return img[y:y + h, x:x + w]