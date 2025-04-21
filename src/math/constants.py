import numpy as np

# angle covered by each sector of the dartboard
WIDTH = 2 * np.pi / 20

# dartboard ring radii (mm)
BULL_INNER = 6.35
BULL_OUTER = 15.9
TRIP_INNER = 99
TRIP_OUTER = 107
DOUB_INNER = 162
DOUB_OUTER = 170

SECTOR_VALUES = [6, 13, 4, 18, 1, 20, 5, 12, 9, 14, 11, 8, 16, 7, 19, 3, 17, 2, 15, 10]


MM_PER_INCH = 25.4
SCALE_FACTOR = 0.4 / MM_PER_INCH  # for converting from dartboard dimensions to plotting dimensions
PAD_WIDTH = 0.1

HEATMAP_PAD_MM = 25

SECTOR_PDF_IGNORE_THRESHOLD = 1e-10

DPI = 200

C_MAP = 'gist_heat'