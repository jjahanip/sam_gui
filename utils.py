import cv2
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

MAX_WIN_SIZE = 1000


def select_file():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()
    return file_path


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def adjust_histogram_gui(image):
    # Calculate the size of the window based on the size of the trackbars
    trackbar_height = 50
    window_width = min(image.shape[1], MAX_WIN_SIZE)
    window_height = min(image.shape[0] + 2 * trackbar_height, MAX_WIN_SIZE)

    # Create a window to display the histogram and trackbars
    cv2.namedWindow('Adjust Histogram', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Adjust Histogram', window_width, window_height)

    # Bring the window to the front
    cv2.setWindowProperty('Adjust Histogram', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

    # Create a canvas to draw the histogram and trackbars
    canvas = np.zeros((window_height, window_width, 3), np.uint8)

    # Create trackbars for adjusting the knobs
    cv2.createTrackbar('Min', 'Adjust Histogram', 0, 255, lambda x: None)
    cv2.createTrackbar('Max', 'Adjust Histogram', 255, 255, lambda x: None)

    while True:
        # Get the current trackbar positions
        min_val = cv2.getTrackbarPos('Min', 'Adjust Histogram')
        max_val = cv2.getTrackbarPos('Max', 'Adjust Histogram')

        # Clip the image values based on the knobs for each channel
        adjusted_image = np.clip(image, min_val, max_val)

        # Normalize the image values between min_val and max_val for each channel
        adjusted_image = np.interp(adjusted_image, [min_val, max_val], [0, 255])

        # Convert the adjusted image to 8-bit unsigned integers
        adjusted_image = adjusted_image.astype(np.uint8)

        # Display the canvas
        cv2.imshow('Adjust Histogram', canvas)

        # Display the adjusted image
        cv2.imshow('Adjust Histogram', cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB))

        # Wait for key press (press 'q' to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Adjust Histogram', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Release the window and resources
    cv2.destroyAllWindows()

    return adjusted_image


def annotate_image_gui(image, mode='point'):
    """
    Annotate an image with points and boxes

    # Usage example
    # right click - annotate a point with label 0 (background)
    # left click - annotate a point with label 1 (foreground)
    # left click + drag - annotate a box
    # scroll wheel - zoom in or out

    # keyboard shortcuts:
    # 'b' - switch to box annotation mode
    # 'p' - switch to point annotation mode
    # 'q' - quit
    """

    # Create a copy of the image to draw annotations on
    annotated_image = image.copy()

    # Create lists to store the annotated points and boxes
    points = []
    boxes = []
    point_labels = []
    unscaled_points = []  # To store unscaled points
    unscaled_boxes = []  # To store unscaled boxes

    # Flag to indicate annotation mode (True for points, False for boxes)
    point_mode = True if mode == 'point' else False


    # Zoom parameters
    zoom_factor = 1.0
    zoom_scale = 0.1

    # Mouse callback function
    def annotate_callback(event, x, y, flags, param):
        nonlocal points, boxes, point_labels, unscaled_points, unscaled_boxes, annotated_image, point_mode, zoom_factor

        if event == cv2.EVENT_LBUTTONDOWN:
            # Add clicked point to the list if in point annotation mode
            if point_mode:
                unscaled_points.append((x, y))
                points.append((int(x / zoom_factor), int(y / zoom_factor)))  # Scale the point and store
                # Draw the annotated point on the image
                cv2.circle(annotated_image, (x, y), 5, (0, 255, 0), -1)
                cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

                point_labels.append(1)  # Add the label 1 to the point_labels list

            # Start recording box annotation if in box annotation mode
            else:
                points.append((x, y))
                unscaled_points.append((int(x / zoom_factor), int(y / zoom_factor)))  # Scale the point and store

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Add clicked point to the list with label 0
            unscaled_points.append((x, y))
            points.append((int(x / zoom_factor), int(y / zoom_factor)))  # Scale the point and store
            # Draw the annotated point on the image
            cv2.circle(annotated_image, (x, y), 5, (255, 0, 0), -1)
            cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

            point_labels.append(0)  # Add the label 0 to the point_labels list

        elif event == cv2.EVENT_LBUTTONUP:
            # Add box coordinates to the list if in box annotation mode
            if not point_mode:
                xmin = min(points[-1][0], x)
                ymin = min(points[-1][1], y)
                xmax = max(points[-1][0], x)
                ymax = max(points[-1][1], y)
                unscaled_boxes.append([xmin, ymin, xmax, ymax])  # Store unscaled box
                boxes.append(
                    [int(coord / zoom_factor) for coord in [xmin, ymin, xmax, ymax]])  # Scale the box and store

                # Draw the annotated box on the image
                cv2.rectangle(annotated_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

                # Remove the last point from the points if it was used to draw a box
                points.pop()
                unscaled_points.pop()

        elif event == cv2.EVENT_MOUSEWHEEL:
            # Zoom in or out based on the mouse wheel movement
            if flags > 0:
                zoom_factor += zoom_scale
            else:
                zoom_factor -= zoom_scale

            # Apply the zoom transformation to the image and update the displayed image
            zoomed_image = cv2.resize(image, None, fx=zoom_factor, fy=zoom_factor, interpolation=cv2.INTER_LINEAR)
            annotated_image = zoomed_image.copy()

            # Adjust the coordinates of the points and boxes based on the zoom factor
            adjusted_points = [(int(x * zoom_factor), int(y * zoom_factor)) for x, y in points]
            adjusted_boxes = [[int(coord * zoom_factor) for coord in box] for box in boxes]

            # Redraw the points and boxes on the zoomed image
            for point in adjusted_points:
                cv2.circle(annotated_image, point, 5, (0, 255, 0), -1)
            for box in adjusted_boxes:
                cv2.rectangle(annotated_image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            cv2.imshow('Annotated Image', cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))

    # Create a named window and set the mouse callback
    cv2.namedWindow('Annotated Image')
    cv2.setMouseCallback('Annotated Image', annotate_callback)

    # Display the initial image
    cv2.imshow('Annotated Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    while True:
        # Wait for a key press
        key = cv2.waitKey(1) & 0xFF

        # Switch to box annotation mode if 'b' is pressed
        if key == ord('b'):
            point_mode = False
            print('Switched to box annotation mode')

        # Switch to point annotation mode if 'p' is pressed
        elif key == ord('p'):
            point_mode = True
            print('Switched to point annotation mode')

        # Quit if 'q' is pressed
        elif key == ord('q') or cv2.getWindowProperty('Annotated Image', cv2.WND_PROP_VISIBLE) < 1:
            break

    # Close all windows
    cv2.destroyAllWindows()

    # Return the annotated points and boxes
    return points, boxes, point_labels


if __name__ == '__main__':
    ############# adjust_histogram_gui function #############
    # Read an image
    image = cv2.imread('imgs/img1.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change BGR to RGB

    # Adjust the histogram
    adjusted_image = adjust_histogram_gui(image)
    ############# adjust_histogram_gui function #############

    ############# annotate_image_gui function #############
    # Read an image
    image = cv2.imread('imgs/img1.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # change BGR to RGB

    # Adjust the histogram
    points, boxes, point_labels = annotate_image_gui(image)
    ############# annotate_image_gui function #############
