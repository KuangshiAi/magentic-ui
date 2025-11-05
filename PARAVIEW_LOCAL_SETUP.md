# Local ParaView Setup Guide

This guide shows how to run the ParaView Agent using your local ParaView installation instead of Docker.

## Benefits of Local Mode

- ✅ **No Docker complexity** - No container management
- ✅ **Native performance** - Direct access to GPU/hardware
- ✅ **True multi-client sync** - GUI updates work correctly
- ✅ **Simpler debugging** - Direct process access
- ✅ **Faster startup** - No container overhead

## Prerequisites

1. **Conda environment with ParaView installed**
   ```bash
   conda create -n paraview_mcp python=3.10
   conda activate paraview_mcp
   conda install -c conda-forge paraview
   ```

2. **Verify ParaView installation**
   ```bash
   conda activate paraview_mcp
   which pvserver  # Should show path in conda env
   which paraview  # Should show path in conda env
   ```

## Configuration

Use the `config_paraview_local.yaml` configuration file:

```yaml
# Enable local ParaView mode
paraview_agent_config:
  use_local_paraview: true  # KEY: Enable local mode
  conda_env: "paraview_mcp"  # Your conda environment name
  pvserver_port: 11111
  auto_gui_connect: true
  gui_connect_wait_time: 10
  data_dir: "./data"

  # MCP settings
  paraview_mcp_python_path: "/path/to/miniconda3/envs/paraview_mcp/bin/python"
  paraview_mcp_script_path: "src/paraview_mcp/paraview_mcp_server.py"
```

## Usage

### 1. Start the Application

```bash
python main.py --config config_paraview_local.yaml
```

### 2. What Happens

When you first interact with the ParaView agent:

1. **pvserver starts** in the conda environment on port 11111
2. **ParaView GUI opens** automatically on your desktop
3. **GUI auto-connects** to pvserver (10 second wait)
4. **MCP connection** is established to pvserver
5. Ready to visualize!

### 3. Using the ParaView Agent

```
User: Load the bonsai dataset
```

The agent will:
- Load `/path/to/data/bonsai_256x256x256_uint8.raw`
- Display should appear **immediately in the ParaView GUI window**
- No manual interaction needed!

### 4. Cleanup

The ParaView processes (pvserver and GUI) will automatically stop when:
- You close the application
- The session ends
- Or manually: The agent's `close()` method is called

## Troubleshooting

### GUI doesn't open

**Check if ParaView is accessible:**
```bash
conda activate paraview_mcp
paraview --version
```

**Check the auto-connect script:**
- Located in `/tmp/paraview_autoconnect_*.py`
- Review logs for connection errors

### MCP can't connect to pvserver

**Verify pvserver is running:**
```bash
ps aux | grep pvserver
netstat -an | grep 11111
```

**Check logs:**
```bash
tail -f ~/paraview_logs/paraview_mcp_external.log
```

### GUI doesn't sync with MCP

**This should NOT happen in local mode!**

If you still see sync issues:
1. Check that `use_local_paraview: true` in config
2. Verify only ONE pvserver process is running
3. Check GUI connected (should see connection in GUI title bar)
4. Review logs for errors

## Architecture

### Local Mode Flow

```
Application Start
    ↓
ParaViewLocalManager.start()
    ↓
├─→ Start pvserver (conda env)
│       - Port: 11111
│       - Mode: --multi-clients
│
├─→ Wait for pvserver ready
│       - Socket check on port 11111
│
├─→ Start ParaView GUI
│       - Runs auto_connect.py script
│       - Connects to localhost:11111
│       - Creates RenderView
│       - Waits 10 seconds
│
└─→ MCP Server connects
        - Same pvserver (localhost:11111)
        - Shares GUI's session
        - Operations sync immediately!
```

### vs Docker Mode Flow

```
Application Start
    ↓
Docker Container Start
    ↓
├─→ Start Xvfb (virtual display)
├─→ Start VNC server
├─→ Start noVNC (web interface)
├─→ Start pvserver
├─→ Start ParaView GUI (in container)
└─→ MCP connects from HOST
        - Through Docker network
        - Separate client session
        - Sync issues! ❌
```

## File Paths

### Important Files

- **Local Manager**: `src/magentic_ui/tools/paraview_local/paraview_local_manager.py`
- **Agent**: `src/magentic_ui/agents/paraview/_agent.py`
- **Config**: `src/magentic_ui/agents/paraview/_config.py`
- **MCP Server**: `src/paraview_mcp/paraview_mcp_server.py`
- **Manager**: `src/paraview_mcp/paraview_manager.py`

### Configuration Files

- **Local Mode**: `config_paraview_local.yaml` (NEW!)
- **Docker Mode**: `config_paraview_example.yaml` (Legacy)

## Comparison

| Feature | Local Mode | Docker Mode |
|---------|-----------|-------------|
| Setup Complexity | Low | High |
| GUI Sync | ✅ Works | ❌ Broken |
| Performance | Native | Virtualized |
| GPU Access | Direct | Limited |
| Portability | Requires local install | Fully portable |
| Debugging | Easy | Complex |
| Multi-client | Native support | Limited |

## Recommended Setup

**For Development/Research**: Use **Local Mode**
- Faster iteration
- Better debugging
- Reliable GUI sync

**For Deployment/Sharing**: Use **Docker Mode**
- Consistent environment
- No local dependencies
- Easier distribution

## Next Steps

1. Copy `config_paraview_local.yaml` to `config.yaml`
2. Update paths in the config
3. Run: `python main.sh`
4. Enjoy working ParaView visualizations! 🎉
