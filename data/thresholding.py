import time
import cv2
import numpy as np


def write_sdp(file_path, pts):
    try:
        with open(file_path, "w") as file:
            for point in pts:
                file.write(f"{point[0]} {point[1]}\n")
    except Exception as e:
        print(f"Error while writing SDP file: {e}")


def read_sdp(file_path):
    try:
        with open(file_path, "r") as file:
            print(f"Reading {file_path} file...")
            lines = [
                line.strip()
                for line in file.readlines()
                if line.strip() and not line.startswith("#")
            ]
            points = [tuple(map(float, line.split())) for line in lines]
            return points
    except Exception as e:
        print(f"Error reading file: {e}")


def threshold_img_to_sdp(img_path, output_sdp_path):
    process = 0
    start = time.time()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    closed_image = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=5)
    closed_image = cv2.bitwise_not(closed_image)
    contours, _ = cv2.findContours(
        closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    points = [tuple(ctr) for ctr in largest_contour[:, 0, :]]
    end = time.time()
    process = (end - start) * 1000
    write_sdp(output_sdp_path, points)
    return process
