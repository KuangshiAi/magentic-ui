#!/usr/bin/env bash
set -euo pipefail

# --- Display / geometry (virtual desktop) ---
: "${DISPLAY:=:1}"
: "${WIDTH:=1280}"
: "${HEIGHT:=800}"

# --- pvserver config ---
: "${PV_HOST:=localhost}"
: "${PV_PORT:=11111}"
# CRITICAL: --enable-bt enables collaboration mode for multi-client synchronization
: "${PVSERVER_OPTS:=--multi-clients --enable-bt}"

# --- GUI auto-connect (set AUTO_GUI_CONNECT=0 to skip launching ParaView GUI) ---
: "${AUTO_GUI_CONNECT:=1}"

# --- Prepare runtime dirs (X11 + XDG) ---
mkdir -p /tmp/.X11-unix
chmod 1777 /tmp/.X11-unix
mkdir -p "${XDG_RUNTIME_DIR:-/tmp/xdg}"
chmod 700 "${XDG_RUNTIME_DIR:-/tmp/xdg}"

# --- Start virtual framebuffer (headless X) ---
Xvfb "${DISPLAY}" -screen 0 "${WIDTH}x${HEIGHT}x24" -nolisten tcp &
sleep 0.5

# --- Window manager (lightweight) ---
openbox &

# --- VNC server (expose the Xvfb display) ---
x11vnc -display "${DISPLAY}" -forever -shared -rfbport 5900 -nopw -repeat -rfbversion 3.8 -noxdamage &

# --- noVNC proxy (HTML5 VNC on :6080) ---
/opt/noVNC/utils/novnc_proxy --vnc localhost:5900 --listen 0.0.0.0:6080 &

# --- Start pvserver (multi-clients) ---
echo "[$(date)] Starting pvserver on port ${PV_PORT}..." >> "$HOME/startup.log"
pvserver --server-port="${PV_PORT}" ${PVSERVER_OPTS} &> "$HOME/pvserver.log" &
PVSERVER_PID=$!
echo "[$(date)] pvserver started with PID ${PVSERVER_PID}" >> "$HOME/startup.log"

# --- Wait until pvserver is reachable ---
echo "[$(date)] Waiting for pvserver to be ready..." >> "$HOME/startup.log"
for i in {1..60}; do
  if nc -z "${PV_HOST}" "${PV_PORT}" 2>/dev/null; then
    echo "[$(date)] pvserver is ready and accepting connections" >> "$HOME/startup.log"
    break
  fi
  sleep 0.5
done

# --- Optionally start ParaView GUI and auto-connect to pvserver ---
if [ "${AUTO_GUI_CONNECT}" = "1" ]; then
  echo "[$(date)] Starting ParaView GUI with auto-connect..." >> "$HOME/startup.log"
  # Set environment variable to disable welcome dialog
  export PV_DISABLE_WELCOME_DIALOG=1

  # CRITICAL FIX: Use --server-url instead of --script for proper client registration
  # This ensures GUI connects as a proper collaborative client, not through a script context
  paraview --server-url="cs://${PV_HOST}:${PV_PORT}" &> "$HOME/paraview-gui.log" &
  GUI_PID=$!
  echo "[$(date)] ParaView GUI started with PID ${GUI_PID}" >> "$HOME/startup.log"
  echo "[$(date)] GUI connecting to cs://${PV_HOST}:${PV_PORT}" >> "$HOME/startup.log"

  # Give GUI extra time to establish connection and create view
  sleep 2
  echo "[$(date)] GUI initialization complete, ready for MCP connections" >> "$HOME/startup.log"
else
  echo "[$(date)] AUTO_GUI_CONNECT disabled, skipping GUI startup" >> "$HOME/startup.log"
fi

# --- Convenience terminal in the virtual desktop ---
xterm -fa Monospace -fs 11 -hold -e bash -l &

# Keep container alive while any bg process is running
wait -n || true
