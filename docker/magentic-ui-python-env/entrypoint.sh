#!/usr/bin/env bash
set -e

umask 000

# Start Xvfb (X virtual framebuffer) for offscreen rendering
# This provides an X display for ParaView/VTK rendering operations
if [ -z "$XVFB_RUNNING" ]; then
    export XVFB_RUNNING=1
    # Start Xvfb on the display specified by $DISPLAY (default :99)
    Xvfb ${DISPLAY} -screen 0 1280x1024x24 -nolisten tcp &
    XVFB_PID=$!
    # Give Xvfb a moment to start
    sleep 1
    echo "Started Xvfb on display ${DISPLAY} with PID ${XVFB_PID}"
fi

exec "$@"