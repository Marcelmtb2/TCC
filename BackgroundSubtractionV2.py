import cv2
import numpy as np

"""
Object Tracking and Detection Module V2

This module implements the necessary tasks to track and detect
the presence of one object in a video stream, in a controlled environment.

The module has the tools to track the movement of objects and evaluate if
an object brought into the scene is stand still, if it is centered in the
scene by not reaching the image margins, and if it may be occluded by
obstacles or by the hands of a technician operating the system.

"""


# Initialize the object capture system
def initialize_bg_sub():
    """
    Configure the background subtractor with default values.

    Returns:
        The configured background subtractor object.
    """
    # Initialize the background subtractor with default values
    fgbgMOG2 = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=50, detectShadows=True
    )
    # TODO: test values and explain their meaning

    # All default values were confirmed, as testing history and varThreshold
    # did not alter much the final result
    return fgbgMOG2


def is_object_at_image(image, show_overlay=False):
    """
    Check if there are contours of an object in the image.

    Args:
        image: The preprocessed resized grayscaled image.
        show_overlay (bool): Flag to show overlay images for debugging.

    Returns:
        list: A list containing a boolean indicating if an object is
        present and the contours.
    """
    # Check if the first frame contains contours of an object.
    # Configure the background subtraction method with a very high
    # learning rate to quickly adapt the average background until
    # the object is removed.
    # To detect edges, use thresholding on grayscale images or apply
    # the Canny filter to highlight contour edges
    # Receives preprocessed image

    # Discover how to make dynamic thresholding!
    # To overcome low light conditions
    # HSV Analysis?

    # If the object is darker than the green background, in grayscale
    thresh_dark = cv2.threshold(image, 90, 255, cv2.THRESH_BINARY_INV)[1]

    # If the object is lighter than the green background, in grayscale
    thresh_light = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]

    # The final mask is the bitwise OR combination of the two masks. Hysteresis
    # tries to avoid shadows on the green sheet.
    thresh = cv2.bitwise_or(thresh_dark, thresh_light)

    # If any contour is identified, return true
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    if len(contours):
        if show_overlay:
            # Copy image to avoid modifying it
            output_image = image.copy()

            # Show binary masks
            cv2.imshow("dark mask", thresh_dark)
            cv2.imshow("light mask", thresh_light)
            cv2.imshow("light OR dark mask", thresh)

            # Draw all contours found
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(output_image, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)
            # Visualize the binary image
            cv2.imshow("Binary image", output_image)
            cv2.waitKey(10)
        return [True, contours]
    return [False, contours]
    # Find contours of objects before background subtraction. If there are any
    # contours, max out the learningRate parameter to 0.1 until no object is
    # present in the image


def image_rotation_compare(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    _, thresh1 = cv2.threshold(gray1, 50, 255, cv2.THRESH_BINARY)
    contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    _, thresh2 = cv2.threshold(gray2, 50, 255, cv2.THRESH_BINARY)
    contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not (contours1 or contours2):
        return None

    largest_contour1 = max(contours1, key=cv2.contourArea)
    moments1 = cv2.moments(largest_contour1)

    largest_contour2 = max(contours2, key=cv2.contourArea)
    moments2 = cv2.moments(largest_contour2)

    if moments1["mu20"] + moments1["mu02"] == 0:
        return None
    
    if moments2["mu20"] + moments2["mu02"] == 0:
        return None

    angle1 = 0.5 * np.arctan2(2 * moments1["mu11"], moments1["mu20"] - moments1["mu02"])

    angle2 = 0.5 * np.arctan2(2 * moments2["mu11"], moments2["mu20"] - moments2["mu02"])

    return np.degrees(abs(angle1 - angle2))  # Converter de radianos para graus


def preprocess_image(image, show_overlay=False):
    """
    Preprocess the image by filtering noise and resizing.

    Args:
        image: The original input image.
        show_overlay (bool): Flag to show overlay images for debugging.

    Returns:
        The preprocessed grayscale image.
    """
    # Filter camera scanning noise.
    # Filters tested for standard light conditions

    # Image is in 16:9 format?
    # Considering the following:
    # the camera support mount is being used
    # The BRIO camera image comes in 4k (2160 x 3840)
    height, width = image.shape[0:2]
    # Crop the image to remove the feet of the camera stand
    cropleft = int(0.21875 * width)
    cropright = int(0.77 * width)  # To crop the tip of the stand.
    # 0.78125 was the original value
    # Crop the base of the stand.
    frame = image[0:height, cropleft:cropright]

    # Resize to speed up processing
    reduced = cv2.resize(frame, None, fx=0.5, fy=0.5,
                         interpolation=cv2.INTER_AREA)

    # Filtering camera scanning noise
    # TODO: test values and explain their meaning
    f_bilateral = cv2.bilateralFilter(reduced, d=9, sigmaColor=75,
                                      sigmaSpace=125)
    
    # TODO: Create images of gaussian blur, median filter, and bilateral filter
    # f_bilateral = cv2.GaussianBlur()

    # Convert to grayscale
    gray_frame = cv2.cvtColor(f_bilateral, cv2.COLOR_BGR2GRAY)

    # Debug - see the original image before filtering
    if show_overlay:
        cv2.imshow("Cropped and resized image", reduced)
        cv2.waitKey(10)
    return gray_frame


def find_foreground_object(fgbgMOG2, image, learning_rate=0.0001):
    """
    Identify objects moving relative to the background.

    Args:
        image: The preprocessed image.
        learning_rate (float): The learning rate for the background subtractor.

    Returns:
        The cleaned binary mask of the foreground object.
    """
    # Identify objects moving relative to the background
    # The learningRate parameter controls how long a moving object
    # that stops in the scene will be considered part of the background
    # With a very small learningRate, 0.0001, to preserve the object
    # for some time after stopping the movement. Calibrate so that the object
    # is considered background after 10 seconds, at initialization.
    fgmaskMOG2 = fgbgMOG2.apply(image, learningRate=learning_rate)
    thresh_mask = cv2.threshold(fgmaskMOG2, 210, 255, cv2.THRESH_BINARY)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    clean_mask = cv2.dilate(thresh_mask, kernel, iterations=4)
    return clean_mask


def identify_contours(object_found_binary_mask):
    """
    Locate contours of the possible object in the binary mask.

    Args:
        object_found_binary_mask: The binary mask of the foreground object.

    Returns:
        tuple: A tuple containing lists of valid contours, border contours,
        and the final object bounding box.
    """
    # Locate contours of the possible object
    # Generally, it will not be possible to perfectly identify the object only
    # with background subtraction
    # If there is background subtraction activity, classify if the modified
    # pixels form objects.
    # Discard contours that are within a margin of 5% of the border dimensions
    # Group valid contours into a larger bounding box
    # Confirm the mask calculated with the background subtractor with the
    # object contour from the original image.
    # Returns True or False for detected object or not

    # Detect contours in the calculated mask
    mask_height, mask_width = object_found_binary_mask.shape
    size_border_factor = 0.01  # Minimum distance from the borders of 10 pixels
    margin_top_bot = int(mask_height * size_border_factor)
    margin_left_right = int(mask_width * size_border_factor)
    # Make this border as a percentage, as 1% of the H or W dimension?
    valid_contours = []
    border_contours = []
    final_object_box = ()
    # Detect contours in the calculated mask
    contours = cv2.findContours(
        object_found_binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[0]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # Check if each contour is far from the image borders
        top_left_away_border = (x > margin_left_right) and (y > margin_top_bot)

        top_right_away_border = (((x + w) < (mask_width - margin_left_right))
                                 and (y > margin_top_bot))

        bot_left_away_border = ((x > margin_left_right)
                                and
                                ((y + h) < (mask_height - margin_top_bot)))

        bot_right_away_border = (((x + w) < (mask_width - margin_left_right))
                                 and
                                 ((y + h) < (mask_height - margin_top_bot)))

        image_away_borders = (
            top_left_away_border
            and top_right_away_border
            and bot_left_away_border
            and bot_right_away_border
        )  # True if away

        if image_away_borders:
            valid_contours.append((x, y, w, h))
        else:
            border_contours.append((x, y, w, h))

    # Create a bigger box around bounding boxes if there are valid contours
    if len(valid_contours):
        # Group nearby bounding boxes
        x_min = min([x for (x, y, w_box, h_box) in valid_contours])
        y_min = min([y for (x, y, w_box, h_box) in valid_contours])
        x_max = max([x + w_box for (x, y, w_box, h_box) in valid_contours])
        y_max = max([y + h_box for (x, y, w_box, h_box) in valid_contours])

        # Final bounding box of the estimated object
        final_object_box = (x_min, y_min, x_max - x_min, y_max - y_min)

    return valid_contours, border_contours, final_object_box


def locate_object(fgbgMOG2, image, learning_rate=0.0001):
    """
    Locate the object in the preprocessed image.

    Args:
        image: The input image.
        learning_rate (float): The learning rate for the background subtractor.

    Returns:
        tuple: A tuple containing lists of valid contours, border contours,
        and the final object bounding box.
    """
    # Receives preprocessed image, returns mask and bounding box of the object
    preproc_image = preprocess_image(image)

    clean_mask = find_foreground_object(fgbgMOG2, preproc_image, learning_rate)

    (valid_boxes,
     border_boxes,
     final_object_box) = identify_contours(clean_mask)

    return valid_boxes, border_boxes, final_object_box


if __name__ == "__main__":

    # Original videos recorded in 4k. Reduce resolution for the copy used
    # to calculate contours? Re-encoded videos to Full HD to test
    # the algorithms, then use the original 4k videos

    folder = "video\\videosMock\\"

    # Complete videos for final module testing
    # Simple video
    # cap = cv2.VideoCapture(r"video\teste.mp4")

    # Complex video with various adverse conditions
    # cap = cv2.VideoCapture(r"video\video.mp4")

    # Generic objects
    object1 = "cabo luz off.mp4"
    object2 = "cabo movimento maos luz on.mp4"
    object3 = "caixa clara movimento maos luz on.mp4"
    object4 = "caixa desde inicio luz on.mp4"
    object5 = "caixa luz off.mp4"
    object6 = "caixa mudanca iluminacao.mp4"
    object7 = "Paquimetro luz off.mp4"
    object8 = "Paquimetro mao presente luz off.mp4"
    object9 = "Paquimetro para caixa luz off.mp4"
    object10 = "Regua luz off.mp4"
    object11 = "regua refletiva luz on.mp4"

    # Mock instrument
    object12 = "BaixaIluminacao100luxSombraForte.mp4"
    object13 = "TrocaObjetosAutofocoAtrapalha.mp4"
    object14 = "Iluminacao800_560lux.mp4"
    object15 = "Objeto15segs.mp4"
    object16 = "Objeto15segSubstituido.mp4"
    object17 = "objeto3segs.mp4"
    object18 = "ObjetoInicio.mp4"
    object19 = "ObjetoOrtogonalDiagonal.mp4"
    object20 = "ObjetoReposicionado.mp4"
    object21 = "OclusãoMão.mp4"
    object22 = "OclusãoTempMão.mp4"

    # - To generate HSV values of the background
    object23 = "TemperaturaCor3k_9k.mp4"

    # - HSV values of the background
    object24 = "ContrasteTemperaturaCor3k_9k.mp4"

    # Select the object video
    object = object2

    # Consider in the final version that the device should be a camera
    # Initialize image capture
    device = folder + object
    # If using the main camera, comment the previous line and uncomment
    # the following line
    # device = 0
    camera = cv2.VideoCapture(device)
    cap, fgbgMOG2 = initialize_bg_sub(device)

    # Initialize the check for object presence at the beginning of the video.
    # If the first frame of the video has object contours, keep
    # the object_at_start_flag as True
    object_at_start_flag = True

    # _____________
    # Separating code to visualize results during development
    # Get the frame rate (FPS) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    # Loop to check for object presence frame by frame
    while cap.isOpened():

        ret, frame = cap.read()
        if not ret:
            print("Camera did not send images. Check the equipment.")
            break

        # Preprocess the image by filtering noise and resizing.
        preproc_image = preprocess_image(frame)

        # Checking for an object from the beginning will be the last activity!!
        # Check if the first frame contains contours of an object.
        if object_at_start_flag:
            object_at_start_flag = is_object_at_image(preproc_image)[0]
            learning_rate = 0.1
            # In the first program loop, consider that there is an object
            # present in the image, setting the learningRate to a high value,
            # greater than 0.01. Then, test if there is no contour of
            # any object in the first received image.
            # If there is an object, keep the high rate.
        else:
            learning_rate = 0.0001
        # If it returns False, keep the learning rate low to
        # increase the persistence of the object in the calculated mask

        clean_mask = find_foreground_object(fgbgMOG2, preproc_image,
                                            learning_rate)

        temp_a, temp_b, temp_c = identify_contours(clean_mask)

        valid_boxes, border_boxes, final_object_box = temp_a, temp_b, temp_c
        # Redundancy necessary!
        # As valid objects are sometimes not recognized with a single
        # contour. It was necessary to combine the coordinates of
        # all bounding boxes to find the expected region for the
        # object.
        # Thus, when a final_object_box is found, pass
        # to contour identification in the original image
        if valid_boxes and not border_boxes:
            mask_MOG2 = np.zeros_like(clean_mask)
            x, y, w, h = final_object_box
            cv2.rectangle(mask_MOG2, (x, y), (x + w, y + h), 255,
                          thickness=cv2.FILLED)
            # cv2.imshow("maskMOG2", mask_MOG2)
            mask_object = np.zeros_like(clean_mask)
            # using [1] to get the contours
            instant_contours = is_object_at_image(preproc_image)[1]
            # Receives the contours output
            if instant_contours:  # There may be a case where
                # there are valid_boxes with only one contour and no
                # contour identified in the original image
                # The rest of the process only exists if there are
                # both masks
                x, y, w, h = cv2.boundingRect(np.vstack(instant_contours))
                cv2.rectangle(
                    mask_object, (x, y), (x + w, y + h), 255,
                    thickness=cv2.FILLED)
                # cv2.imshow("objectContour", mask_object)
                intersection = cv2.bitwise_and(mask_MOG2, mask_object)
                union = cv2.bitwise_or(mask_MOG2, mask_object)

                intersection_area = np.sum(intersection > 0)
                union_area = np.sum(union > 0)

                iou = intersection_area / union_area
                # Calculate the Jaccard index (IoU)
                # Set a threshold to consider a good match
                threshold = 0.8
                print(iou)
                if iou > threshold:
                    print("Contour and mask match!")
                else:
                    print("Significant difference between contour and mask.")

        # _____________________________
        # Drawing is not part of the final version

        # Draw valid contours in purple, final object box in green, and
        # border contours in red

        # Visualization will not be part of the final version

        clean_mask_3ch = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
        preproc_image_3ch = cv2.cvtColor(preproc_image, cv2.COLOR_GRAY2BGR)
        if len(valid_boxes):
            cv2.rectangle(preproc_image_3ch, final_object_box, (0, 255, 0), 5)
            cv2.rectangle(clean_mask_3ch, final_object_box, (0, 255, 0), 5)
            for ok_contour in valid_boxes:
                x, y, w, h = ok_contour
                cv2.rectangle(
                    preproc_image_3ch, (x, y), (x + w, y + h),
                    (255, 0, 255), 2)
                cv2.rectangle(clean_mask_3ch, (x, y), (x + w, y + h),
                              (255, 0, 255), 2)
        if len(border_boxes):
            for bad_contour in border_boxes:
                x, y, w, h = bad_contour
                cv2.rectangle(preproc_image_3ch, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)
                cv2.rectangle(clean_mask_3ch, (x, y), (x + w, y + h),
                              (0, 0, 255), 2)

        # Calculate the elapsed time in seconds
        time_elapsed = frame_count / fps
        minutes = int(time_elapsed // 60)
        seconds = int(time_elapsed % 60)

        # Add the time as an overlay on the frame
        overlay_text = f"Time: {minutes:02}:{seconds:02}"
        # cv2.putText(preproc_image_3ch, overlay_text, (10, 30),
        #            cv2.FONT_HERSHEY_SIMPLEX,
        #            1, (0, 255, 255), 2)

        combined_view = cv2.hconcat([preproc_image_3ch, clean_mask_3ch])
        # clean_mask_3ch

        # _____________________
        # Separating code to visualize results during development

        # Debug - visualize filtered grayscale image
        cv2.imshow("img", preproc_image)

        cv2.imshow("Preprocessed image (Left) vs Object Mask (Right)",
                   combined_view)

        frame_count += 1
        # End with 'q' key
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord("q"):
            break

    # print(frame.shape)
    cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
