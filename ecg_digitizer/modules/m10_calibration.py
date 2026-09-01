import numpy as np

def calibrate_signal(pixel_signal, metadata=None):
    """
    Calibrates a pixel signal to time and voltage IF metadata is present.
    DO NOT mock calibration values if they are missing.
    """
    w = len(pixel_signal)
    x_pixel = np.arange(w)
    
    if metadata is None or 'pixels_per_second' not in metadata or 'pixels_per_mv' not in metadata:
        # Strictly no hallucinated calibration
        return {
            "x_pixel": x_pixel,
            "time_seconds": np.full(w, np.nan),
            "voltage_mv": np.full(w, np.nan),
            "status": "uncalibrated"
        }
        
    pixels_per_second = metadata['pixels_per_second']
    pixels_per_mv = metadata['pixels_per_mv']
    
    baseline = np.nanmedian(pixel_signal)
    voltage_mv = (pixel_signal - baseline) / pixels_per_mv
    time_sec = x_pixel / pixels_per_second
    
    return {
        "x_pixel": x_pixel,
        "time_seconds": time_sec,
        "voltage_mv": voltage_mv,
        "status": "calibrated"
    }

def format_biomarker(pixel_value, metadata=None, conversion_factor_key=None):
    """
    Formats a biomarker like QRS duration:
    QRS_duration:
        pixel_value: 18
        physical_value: null
        status: uncalibrated
    """
    if metadata and conversion_factor_key in metadata:
        physical_value = pixel_value / metadata[conversion_factor_key]
        status = "calibrated"
    else:
        physical_value = None
        status = "uncalibrated"
        
    return {
        "pixel_value": float(pixel_value) if pixel_value is not None else None,
        "physical_value": float(physical_value) if physical_value is not None else None,
        "status": status
    }
