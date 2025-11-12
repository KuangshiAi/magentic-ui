# ParaView Docker Script Execution with Server Logging

Execute ParaView Python scripts on a pvserver running in Docker and capture both client and server-side logs/errors.

## Files

- **`execute_with_server_logs.py`** - Main execution script that captures logs from Docker
- **`example_volume_script.py`** - Example script that loads and visualizes volume data

## Usage

### Basic Usage

```bash
python execute_with_server_logs.py \
    --script example_volume_script.py \
    --container <your_container_name>
```

### With Custom Host/Port

```bash
python execute_with_server_logs.py \
    --script example_volume_script.py \
    --container paraview-server \
    --host localhost \
    --port 11111
```

### Without Docker Log Tailing

```bash
python execute_with_server_logs.py \
    --script example_volume_script.py \
    --container paraview-server \
    --no-logs
```

## What It Does

1. **Checks** if the Docker container is running
2. **Starts** tailing Docker logs in the background
3. **Executes** your ParaView Python script using `pvpython`
4. **Captures** both:
   - Client-side output (from your script)
   - Server-side logs (from Docker container)
5. **Reports** exit code and any errors

## Output Format

The script provides three types of output:

### 1. Server Logs (from Docker)
```
======================================================================
SERVER LOGS (from Docker container)
======================================================================
[DOCKER] Connection accepted from: localhost:54321
[DOCKER] Processing request...
```

### 2. Client Output (from your script)
```
======================================================================
CLIENT OUTPUT
======================================================================
[CLIENT] Connecting to pvserver at localhost:11111...
[CLIENT] ✓ Connected to pvserver
[SERVER] Python version: 3.10.12
[SERVER] ✓ Data loaded successfully!
```

### 3. Client Errors/Warnings
```
======================================================================
CLIENT ERRORS/WARNINGS
======================================================================
(   2.840s) [paraview] vtkSocket.cxx:566 ERR| Socket error...
```

## Requirements

- Python 3.6+
- `pvpython` in PATH
- Docker installed and running
- Container with pvserver running

## Finding Your Container Name

```bash
# List running containers
docker ps

# Or use docker ps with format
docker ps --format 'table {{.Names}}\t{{.Status}}'
```

## Example Script Format

Your ParaView Python script should:

1. Connect to the pvserver
2. Use ParaView simple API
3. Include print statements for progress

```python
from paraview.simple import *

# Connect to pvserver
connection = Connect('localhost', 11111)

# Your ParaView operations
reader = ImageReader(...)
reader.UpdatePipeline()

# etc.
```

## Troubleshooting

### Container not found
```bash
# Check running containers
docker ps

# Check all containers (including stopped)
docker ps -a
```

### Connection refused
```bash
# Check if pvserver is running in container
docker exec <container_name> ps aux | grep pvserver

# Check port mapping
docker port <container_name>
```

### Script timeout
The script has a 30-second timeout. For long-running operations, modify the `timeout` parameter in `execute_with_server_logs.py`:

```python
result = subprocess.run(
    cmd,
    capture_output=True,
    text=True,
    timeout=300  # Change to 300 seconds (5 minutes)
)
```

## Advanced Usage

### Capture Logs to File

```bash
python execute_with_server_logs.py \
    --script example_volume_script.py \
    --container paraview-server \
    2>&1 | tee execution_log.txt
```

### Run Multiple Scripts

```bash
for script in script1.py script2.py script3.py; do
    python execute_with_server_logs.py --script $script --container paraview-server
done
```

### Debug Mode

To see more Docker logs, you can increase the tail size by modifying the script:

```python
# In execute_with_server_logs.py, change:
['docker', 'logs', '-f', '--tail', '50', container_name]
# to:
['docker', 'logs', '-f', '--tail', '200', container_name]
```

## Notes

- Print statements in your script execute on the **client side**
- ParaView operations (like `ImageReader`) execute on the **server side**
- Server errors appear in Docker logs, not in script output
- The script automatically cleans up background threads on exit
