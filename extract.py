import cv2
import numpy as np
import os


def extract_and_save_digits(filepaths, output_dir):
    # Create the output directory if it doesn't already exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i, filepath in enumerate(filepaths):
        # Load the image
        image = cv2.imread(filepath)
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Apply binary thresholding
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours which correspond to the digits
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Loop over contours and save each digit as a separate image
        for j, contour in enumerate(contours):
            # Get the bounding box of the contour
            x, y, w, h = cv2.boundingRect(contour)
            # Crop the digit from the image
            digit = thresh[y : y + h, x : x + w]
            # Save the digit image
            digit_filename = f"{output_dir}/digit_{i}_{j}.png"
            cv2.imwrite(digit_filename, digit)
            print(f"Digit saved: {digit_filename}")


# Filepaths of the images you've uploaded
# 1 to 21 png
# "score\score(1).png",
filepaths = [
    "score\score(1).png",
    "score\score(2).png",
    "score\score(3).png",
    "score\score(4).png",
    "score\score(5).png",
    "score\score(6).png",
    "score\score(7).png",
    "score\score(8).png",
    "score\score(9).png",
    "score\score(10).png",
    "score\score(11).png",
    "score\score(12).png",
    "score\score(13).png",
    "score\score(14).png",
    "score\score(15).png",
    "score\score(16).png",
    "score\score(17).png",
    "score\score(18).png",
    "score\score(19).png",
    "score\score(20).png",
    "score\score(21).png",
]

# Call the function to extract digits and save them as templates
extract_and_save_digits(filepaths, "digits_templates")
