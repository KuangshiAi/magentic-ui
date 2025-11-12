"""
Example script to be executed on pvserver.
NOTE: Connection to pvserver is handled by execute_script.py

Important: Regular Python code (print, os.path, etc.) runs on CLIENT side.
Only ParaView API calls (ImageReader, Show, etc.) run on SERVER side.
"""

import sys
from paraview.simple import *

# Data file path (on the server-side)
data_path = '/home/MCPagent/data/bonsai/data/bonsai_256x256x256_uint8.raw'

print(f"[CLIENT] Attempting to load data from server: {data_path}")
print(f"[CLIENT] Note: File must exist on the pvserver, not locally")

# Create ImageReader - this executes on SERVER
print("[CLIENT] Creating ImageReader on server...")
try:
    reader = ImageReader(registrationName='bonsai')
    reader.FileNames = [data_path]

    # Configure reader for bonsai dataset (256x256x256, uint8)
    reader.DataScalarType = 'unsigned char'
    reader.DataExtent = [0, 255, 0, 255, 0, 255]
    reader.DataSpacing = [1.0, 1.0, 1.0]
    reader.DataOrigin = [0.0, 0.0, 0.0]
    reader.DataByteOrder = 'LittleEndian'

    print("[CLIENT] ✓ Reader configured")

    # Update pipeline to actually read the data - THIS is where file is read on SERVER
    print("[CLIENT] Loading data from server (this may take a moment)...")
    reader.UpdatePipeline()

    # Get data information
    data_info = reader.GetDataInformation()
    bounds = data_info.GetBounds()
    num_points = data_info.GetNumberOfPoints()

    print(f"[CLIENT] ✓ Data loaded successfully from server!")
    print(f"[CLIENT] Data bounds: {bounds}")
    print(f"[CLIENT] Number of points: {num_points:,}")
    print(f"[CLIENT] Expected for 256³: 16,777,216 points")

except Exception as e:
    print(f"[CLIENT] ✗ ERROR: Failed to load data from server")
    print(f"[CLIENT] This usually means the file doesn't exist on the server at: {data_path}")
    print(f"[CLIENT] Error: {e}")
    raise

# Create or get render view
print("[CLIENT] Creating render view...")
renderView = GetActiveViewOrCreate('RenderView')
print("[CLIENT] ✓ Render view ready")

# Create display and show
print("[CLIENT] Creating volume representation...")
display = Show(reader, renderView, 'UniformGridRepresentation')

# Set to volume rendering
print("[CLIENT] Setting representation to Volume...")
display.Representation = 'Volume'

# Setup color mapping
print("[CLIENT] Configuring color transfer function...")
ColorBy(display, ('POINTS', 'ImageFile'))

# Get transfer functions
lut = GetColorTransferFunction('ImageFile')
pwf = GetOpacityTransferFunction('ImageFile')

# Configure opacity for volume rendering
print("[CLIENT] Setting opacity transfer function...")
pwf.Points = [
    0.0, 0.0, 0.5, 0.0,      # Value 0: fully transparent
    128.0, 0.5, 0.5, 0.0,    # Value 128: semi-transparent
    255.0, 1.0, 0.5, 0.0     # Value 255: opaque
]

print("[CLIENT] ✓ Transfer functions configured")

# Reset camera and update view
print("[CLIENT] Updating view...")
renderView.ResetCamera()
renderView.Update()

print("[CLIENT] ✓ Volume visualization completed!")
print("[CLIENT] The volume should now be visible in your ParaView client.")

# Print summary
print(f"\n[CLIENT] === Summary ===")
print(f"[CLIENT] Dataset: bonsai (256³ uint8)")
print(f"[CLIENT] Data bounds: {bounds}")
print(f"[CLIENT] Rendering mode: Volume")
print(f"[CLIENT] Status: SUCCESS")
