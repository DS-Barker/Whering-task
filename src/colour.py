import numpy as np
from classifier import resize_to_max_dimension


def get_dominant_colour(image_path, bbox=None):
    """
    Extract dominant colour using median of central region
    """

    # TODO: A more in depth method of finding the colour of the clothing is needed, the colour at the centre of the box will not always be the main colour
    # TODO: Multiple colours are also not factored in
    # TODO: Improvement for production - colour segmentation approach
    # 1. Apply colour-based multi-level thresholding to find regions of similar colour
    # 2. Exclude edge segments/pixels (likely background/shadows at bbox edges)
    # 3. If multiple distinct colour segments found we can return 'multi-coloured'/ a list of the colours
    # 4. If single dominant segment, return that that colour, averaged over the segmented section.
    # Additional logic to account for many segments found, showing patterend/striped items 
    # This would also hopefully reduce background interference

    # Load image
    img = resize_to_max_dimension(image_path, max_dim=400)
    img_array = np.array(img)
    
    # Crop to bbox if provided
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        # Safety check
        if x2 <= x1 or y2 <= y1:
            return 'unknown'
        img_array = img_array[y1:y2, x1:x2]
    
    # Take central 50% of the region to avoid edge effects
    h, w = img_array.shape[:2]
    margin_h = int(h * 0.25)
    margin_w = int(w * 0.25)
    
    if h > 20 and w > 20:  # Only crop if image is large enough
        central_region = img_array[margin_h:-margin_h, margin_w:-margin_w]
    else:
        central_region = img_array
    
    # Flatten to list of pixels
    pixels = central_region.reshape(-1, 3)
    
    # Check we have pixels
    if len(pixels) < 10:
        return 'unknown'
    
    # Get median RGB
    median_rgb = np.median(pixels, axis=0)
    
    return rgb_to_colour_name(median_rgb)

def rgb_to_colour_name(rgb):
    """
    Map RGB to human colour names
    Simplified version with better thresholds
    """
    r, g, b = rgb
    
    # Calculate metrics
    brightness = (r + g + b) / 3
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    saturation = max_val - min_val
    
    # Achromatic colours (white/black/grey)
    if brightness > 200:
        return 'white'
    elif brightness < 50:
        return 'black'
    elif saturation < 30:  # Low saturation = grey
        return 'grey'
    
    # Use thresholds to separate colours better
    # TODO: More explicit thresholds with less overlap would be much better than the rough thresholding done here - reduces chance of false classifications due to logic order
    
    # Blue family
    if b > r + 20 and b > g + 20:
        if b > 180:
            return 'blue'
        elif saturation > 80:
            return 'blue'
        else:
            return 'navy'
    
    # Red/Pink/Orange family
    elif r > g + 10 and r > b + 10:
        if b > 150 and g > 150:  # High r+g+b but r dominant = pink
            return 'pink'
        elif g > r - 50:  # Red and green similar = orange/brown
            if brightness > 150:
                return 'orange'
            else:
                return 'brown'
        else:  # Pure red
            return 'red'
    
    # Green family
    elif g > r + 20 and g > b + 20:
        return 'green'
    
    # Yellow/Beige (high r+g, low b)
    elif r > 130 and g > 130 and b < 130:
        if saturation < 60:
            return 'beige'
        else:
            return 'yellow'
    
    # Purple (r and b both high, g low)
    elif r > 100 and b > 100 and g < min(r, b) - 20:
        return 'purple'
    
    # Fallback - use whichever channel is highest
    if r > g and r > b:
        return 'red'
    elif g > b:
        return 'green'
    else:
        return 'blue'