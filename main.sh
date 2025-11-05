# Use your custom Python environment image
export MAGENTIC_UI_PYTHON_IMAGE="ghcr.io/microsoft/magentic-ui-python-env:latest"

# Use your custom browser image  
export MAGENTIC_UI_BROWSER_IMAGE="ghcr.io/microsoft/magentic-ui-browser:0.0.1"

# Now run Magentic-UI
magentic-ui --port 8081 --config config.yaml