import cv2
import numpy as np
import pandas as pd


def get_regions(img):
    h, w, _ = img.shape
    return {
        "top_left": img[0:h//3, 0:w//3],
        "top_right": img[0:h//3, 2*w//3:w],
        "bottom_left": img[2*h//3:h, 0:w//3],
        "bottom_right": img[2*h//3:h, 2*w//3:w],
        "center": img[h//3:2*h//3, w//3:2*w//3]
    }


def remove_grid_lines(region):
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY_INV)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    cells_only = cv2.bitwise_and(binary, cv2.bitwise_not(grid_mask))

    return gray, binary, grid_mask, cells_only


def detect_cells_from_blobs(region, min_area=15, max_area=400, circularity_thresh=0.20):
    gray, binary, grid_mask, cells_only = remove_grid_lines(region)

    kernel = np.ones((3, 3), np.uint8)
    cells_only = cv2.morphologyEx(cells_only, cv2.MORPH_OPEN, kernel)
    cells_only = cv2.morphologyEx(cells_only, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cells_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cells = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * area / (perimeter ** 2)
        if circularity < circularity_thresh:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)

        detected_cells.append({
            "x": int(x),
            "y": int(y),
            "r": max(int(radius), 3),
            "area": float(area),
            "circularity": float(circularity)
        })

    return gray, binary, grid_mask, cells_only, detected_cells


def classify_detected_blobs(region, detected_cells, dead_threshold=120):
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)

    live = 0
    dead = 0
    cell_info = []

    for cell in detected_cells:
        x, y, r = cell["x"], cell["y"], cell["r"]

        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.circle(mask, (x, y), r, 255, -1)

        mean_intensity = cv2.mean(gray, mask=mask)[0]

        if mean_intensity < dead_threshold:
            label = "dead"
            dead += 1
        else:
            label = "live"
            live += 1

        cell_info.append({
            "x": x,
            "y": y,
            "r": r,
            "mean_intensity": float(mean_intensity),
            "label": label,
            "area": cell.get("area", 0),
            "circularity": cell.get("circularity", 0)
        })

    return live, dead, cell_info


def analyze_region_a(region, dead_threshold=120):
    gray, binary, grid_mask, cells_only, detected_cells = detect_cells_from_blobs(region)
    live, dead, cell_info = classify_detected_blobs(
        region,
        detected_cells,
        dead_threshold=dead_threshold
    )
    return live, dead, detected_cells, cell_info


def detect_cells_from_rings(region, min_radius=4, max_radius=14):
    gray = cv2.cvtColor(region, cv2.COLOR_RGB2GRAY)
    gray_eq = cv2.equalizeHist(gray)

    _, binary = cv2.threshold(gray_eq, 215, 255, cv2.THRESH_BINARY_INV)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))

    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    grid_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
    cells_only = cv2.bitwise_and(binary, cv2.bitwise_not(grid_mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cells_only = cv2.morphologyEx(cells_only, cv2.MORPH_CLOSE, kernel)
    cells_only = cv2.dilate(cells_only, kernel, iterations=2)

    contours, _ = cv2.findContours(cells_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_cells = []

    for cnt in contours:
        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 18:
            continue

        (x, y), radius = cv2.minEnclosingCircle(cnt)
        if radius < min_radius or radius > max_radius:
            continue

        x_rect, y_rect, w, h = cv2.boundingRect(cnt)
        aspect_ratio = w / h if h > 0 else 0

        if aspect_ratio < 0.6 or aspect_ratio > 1.4:
            continue

        detected_cells.append({
            "x": int(x),
            "y": int(y),
            "r": max(int(radius), 3),
            "perimeter": float(perimeter),
            "aspect_ratio": float(aspect_ratio)
        })

    return detected_cells


def classify_cells_by_color(region, detected_cells):
    live = 0
    dead = 0
    cell_info = []

    for cell in detected_cells:
        x, y, r = cell["x"], cell["y"], cell["r"]
        inner_r = max(int(r * 0.45), 2)

        mask = np.zeros(region.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (x, y), inner_r, 255, -1)

        mean_rgb = cv2.mean(region, mask=mask)

        mean_r = mean_rgb[0]
        mean_g = mean_rgb[1]
        mean_b = mean_rgb[2]

        brightness = (mean_r + mean_g + mean_b) / 3
        blue_green_score = ((mean_g + mean_b) / 2) - mean_r

        if brightness > 100:
            label = "live"
            live += 1
        elif blue_green_score > 8:
            label = "dead"
            dead += 1
        else:
            label = "live"
            live += 1

        cell_info.append({
            "x": x,
            "y": y,
            "r": r,
            "mean_r": float(mean_r),
            "mean_g": float(mean_g),
            "mean_b": float(mean_b),
            "blue_green_score": float(blue_green_score),
            "brightness": float(brightness),
            "label": label
        })

    return live, dead, cell_info


def analyze_region_a_ring_color(region):
    detected_cells = detect_cells_from_rings(region)
    live, dead, cell_info = classify_cells_by_color(region, detected_cells)
    return live, dead, detected_cells, cell_info


def result_is_suspicious(live_count, dead_count, detected_cells, min_total=1, max_total=12):
    total = live_count + dead_count

    if len(detected_cells) == 0:
        return True
    if total == 0:
        return True
    if total < min_total:
        return True
    if total > max_total:
        return True

    return False


def analyze_region_with_fallback(region):
    live_a, dead_a, cells_a, info_a = analyze_region_a(region, dead_threshold=120)

    if not result_is_suspicious(live_a, dead_a, cells_a, min_total=1):
        return {
            "method": "A",
            "live": live_a,
            "dead": dead_a,
            "cells": cells_a,
            "cell_info": info_a,
            "suspicious": False
        }

    live_ring, dead_ring, cells_ring, info_ring = analyze_region_a_ring_color(region)

    if not result_is_suspicious(live_ring, dead_ring, cells_ring, min_total=1):
        return {
            "method": "A_ring_color",
            "live": live_ring,
            "dead": dead_ring,
            "cells": cells_ring,
            "cell_info": info_ring,
            "suspicious": False
        }

    return {
        "method": "A_ring_color",
        "live": live_ring,
        "dead": dead_ring,
        "cells": cells_ring,
        "cell_info": info_ring,
        "suspicious": True
    }


def draw_detected_cells(region, cell_info):
    img_copy = region.copy()

    for cell in cell_info:
        x, y, r = cell["x"], cell["y"], cell["r"]

        if cell["label"] == "dead":
            color = (255, 0, 0)
        else:
            color = (0, 255, 0)

        cv2.circle(img_copy, (x, y), r, color, 2)
        cv2.circle(img_copy, (x, y), 2, color, -1)

    return img_copy


def summarize_ai_counts(results_df):
    total_live = results_df["live_ai"].sum()
    total_dead = results_df["dead_ai"].sum()
    total_cells = total_live + total_dead

    viability = 0 if total_cells == 0 else (total_live / total_cells) * 100

    return {
        "total_live_ai": int(total_live),
        "total_dead_ai": int(total_dead),
        "total_cells_ai": int(total_cells),
        "viability_ai": round(viability, 2)
    }


def compare_manual_vs_ai(ai_summary, manual_live, manual_dead):
    ai_live = ai_summary["total_live_ai"]
    ai_dead = ai_summary["total_dead_ai"]
    ai_viability = ai_summary["viability_ai"]

    manual_total = manual_live + manual_dead
    if manual_total == 0:
        return {
            "flag": "red",
            "message": "Manual total is zero. Re-enter counts."
        }

    manual_viability = (manual_live / manual_total) * 100

    live_diff = abs(ai_live - manual_live)
    dead_diff = abs(ai_dead - manual_dead)
    viability_diff = abs(ai_viability - manual_viability)

    if viability_diff <= 5 and live_diff <= 10 and dead_diff <= 10:
        flag = "green"
        message = "AI and manual counts agree closely."
    elif viability_diff <= 10 and live_diff <= 20 and dead_diff <= 20:
        flag = "yellow"
        message = "AI and manual counts are somewhat different. Double-check recommended."
    else:
        flag = "red"
        message = "AI and manual counts differ strongly. Recount recommended."

    return {
        "flag": flag,
        "message": message,
        "ai_live": ai_live,
        "ai_dead": ai_dead,
        "manual_live": manual_live,
        "manual_dead": manual_dead,
        "ai_viability": round(ai_viability, 2),
        "manual_viability": round(manual_viability, 2),
        "live_difference": live_diff,
        "dead_difference": dead_diff,
        "viability_difference": round(viability_diff, 2)
    }


def check_region_consistency(results_df):
    totals = results_df["total_ai"].tolist()

    if len(totals) < 2:
        return {"flag": "green", "message": "Not enough regions to compare."}

    mean_total = np.mean(totals)
    std_total = np.std(totals)

    if mean_total == 0:
        return {"flag": "red", "message": "No cells detected in any region."}

    cv = (std_total / mean_total) * 100

    if mean_total < 3:
        return {
            "flag": "yellow",
            "message": "Cell counts are low per region, so variation may be less reliable.",
            "mean_total": round(mean_total, 2),
            "std_total": round(std_total, 2),
            "cv_percent": round(cv, 2)
        }

    if cv <= 20:
        flag = "green"
        message = "Counts across regions are consistent."
    elif cv <= 35:
        flag = "yellow"
        message = "Moderate variation across regions. Mixing/counting should be checked."
    else:
        flag = "red"
        message = "High variation across regions. Recount recommended."

    return {
        "flag": flag,
        "message": message,
        "mean_total": round(mean_total, 2),
        "std_total": round(std_total, 2),
        "cv_percent": round(cv, 2)
    }


def final_decision(comparison, consistency):
    flags = [comparison["flag"], consistency["flag"]]

    if "red" in flags:
        final_flag = "red"
        final_message = "Recount recommended."
    elif "yellow" in flags:
        final_flag = "yellow"
        final_message = "Borderline result. Double-check recommended."
    else:
        final_flag = "green"
        final_message = "Counts appear reliable."

    return {
        "final_flag": final_flag,
        "final_message": final_message,
        "comparison_message": comparison["message"],
        "consistency_message": consistency["message"]
    }


def analyze_hemocytometer(image_rgb, manual_live=0, manual_dead=0):
    regions = get_regions(image_rgb)

    results = []
    region_visuals = {}

    for region_name, region_img in regions.items():
        result = analyze_region_with_fallback(region_img)

        live_count = result["live"]
        dead_count = result["dead"]
        cells = result["cells"]
        cell_info = result["cell_info"]

        results.append({
            "region": region_name,
            "live_ai": live_count,
            "dead_ai": dead_count,
            "total_ai": live_count + dead_count,
            "detected_cells": len(cells),
            "method_used": result["method"],
            "suspicious": result["suspicious"]
        })

        region_visuals[region_name] = draw_detected_cells(region_img, cell_info)

    results_df = pd.DataFrame(results)
    summary = summarize_ai_counts(results_df)
    comparison = compare_manual_vs_ai(summary, manual_live, manual_dead)
    consistency = check_region_consistency(results_df)
    decision = final_decision(comparison, consistency)

    return results_df, summary, comparison, consistency, decision, region_visuals