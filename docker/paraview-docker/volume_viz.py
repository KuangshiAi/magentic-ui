#!/usr/bin/env python3
"""
ParaView Python script to load raw data and enable volume visualization.
This script connects to a running pvserver and creates a volume rendering.

Usage:
    pvpython volume_viz.py [--host HOST] [--port PORT] [--debug]

Default connection: localhost:11111
"""

import argparse
import sys
import traceback
from paraview.simple import *

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Load data and create volume visualization')
    parser.add_argument('--host', default='localhost', help='pvserver host (default: localhost)')
    parser.add_argument('--port', type=int, default=11111, help='pvserver port (default: 11111)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    args = parser.parse_args()

    # Enable debug logging if requested
    if args.debug:
        print("DEBUG MODE ENABLED")
        print(f"Python version: {sys.version}")
        print(f"ParaView version: {GetParaViewVersion()}")
        import paraview
        print(f"ParaView module location: {paraview.__file__}")

    print(f"Connecting to pvserver at {args.host}:{args.port}...")

    try:
        # Connect to the pvserver
        connection = Connect(args.host, args.port)

        if connection is None:
            print("ERROR: Failed to connect to pvserver!")
            print(f"Make sure pvserver is running on {args.host}:{args.port}")
            return 1

    except Exception as e:
        print(f"ERROR: Connection failed with exception: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

    print("Connected successfully!")

    if args.debug:
        print(f"Connection info: {connection}")
        print(f"Active connection: {GetActiveConnection()}")

    # Data file path (inside the container)
    data_path = '/home/MCPagent/data/bonsai/data/bonsai_256x256x256_uint8.raw'

    print(f"Loading data from {data_path}...")

    try:
        # Create ImageReader for the raw file
        # For bonsai dataset: 256x256x256, uint8
        reader = ImageReader(registrationName='bonsai')
        reader.FileNames = [data_path]

        # Set the dimensions and data type
        reader.DataScalarType = 'unsigned char'  # uint8
        reader.DataExtent = [0, 255, 0, 255, 0, 255]  # 256x256x256
        reader.DataSpacing = [1.0, 1.0, 1.0]
        reader.DataOrigin = [0.0, 0.0, 0.0]
        reader.DataByteOrder = 'LittleEndian'

        if args.debug:
            print(f"Reader created: {reader}")
            print(f"Reader properties: {dir(reader)}")

        # Try to update the reader to check if file exists
        reader.UpdatePipeline()

        print("Data loaded successfully!")

    except Exception as e:
        print(f"ERROR: Failed to load data file!")
        print(f"Error message: {e}")
        if args.debug:
            traceback.print_exc()
        print(f"\nTroubleshooting:")
        print(f"1. Check if file exists: {data_path}")
        print(f"2. Verify file permissions")
        print(f"3. Ensure correct file path in container")
        return 1

    try:
        # Get active view (or create one if needed)
        renderView = GetActiveViewOrCreate('RenderView')

        if args.debug:
            print(f"Render view: {renderView}")

        # Create volume representation
        print("Creating volume representation...")
        display = Show(reader, renderView, 'UniformGridRepresentation')

        if args.debug:
            print(f"Display object: {display}")
            print(f"Available representations: {display.Representation.Available}")

        # Set representation to Volume
        display.Representation = 'Volume'

        # Set up color transfer function
        ColorBy(display, ('POINTS', 'ImageFile'))

        # Get the color transfer function
        lut = GetColorTransferFunction('ImageFile')

        # Get the opacity transfer function
        pwf = GetOpacityTransferFunction('ImageFile')

        if args.debug:
            print(f"Color transfer function: {lut}")
            print(f"Opacity transfer function: {pwf}")

        # Optional: Set some default opacity mapping for volume rendering
        # This creates a simple ramp - you can adjust these values
        pwf.Points = [0.0, 0.0, 0.5, 0.0,
                      128.0, 0.5, 0.5, 0.0,
                      255.0, 1.0, 0.5, 0.0]

        # Reset camera to fit the data
        renderView.ResetCamera()

        # Update the view
        renderView.Update()

        print("Volume visualization enabled!")
        print("Volume rendering is now active in the connected ParaView client.")

    except Exception as e:
        print(f"ERROR: Failed to create volume visualization!")
        print(f"Error message: {e}")
        if args.debug:
            traceback.print_exc()
        return 1

    # Optionally save a screenshot (if running with offscreen rendering)
    # SaveScreenshot('/tmp/volume_viz.png', renderView, ImageResolution=[1920, 1080])

    print("\nScript completed successfully!")
    print("The volume visualization should now be visible in your ParaView client.")

    return 0

if __name__ == '__main__':
    sys.exit(main())
