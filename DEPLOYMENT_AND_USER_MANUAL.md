# LiDAR Workbench & Multi-VPS Cloud Infrastructure
## Complete Deployment Guide, Linux Transition Manual & Operations Reference
**Version 2.0 LTS — November 2026**

---

## 1. System Architecture Overview

The system is architected into two clearly separated, production-ready tiers:
1. **Open-Source Desktop Client (`lidar-workbench`):** Runs natively on Linux and Windows for point cloud processing, strip adjustment, Pointcept AI ground classification, QGIS project styling, and peer-to-peer compute clustering. Zero server private keys or licensing barriers on Linux.
2. **Closed-Source Cloud Platform & Multi-VPS Storage Pool (`lidar-server`):** Central orchestrator (FastAPI + PostgreSQL) paired with horizontally scalable, low-cost Storage VPS instances (e.g. Hetzner, Netcup, Contabo @ ~13€/TB/month) delivering ultra-fast byte-range COPC point cloud and WebP raster streaming.

| Component | Default Port | Core Stack | Recommended Hardware |
| :--- | :--- | :--- | :--- |
| **Central Cloud Server** | `8000` / `443` | FastAPI, PostgreSQL 16, SQLAlchemy, Jinja2, Uvicorn | 2-4 vCPU, 4-8 GB RAM, 50 GB SSD (Debian/Ubuntu) |
| **Storage VPS Node(s)** | `9000` / `443` | Nginx (sendfile), FastAPI Agent (< 25MB RAM) | 1-2 vCPU, 2-4 GB RAM, 1-2 TB VPS (~13€/mo) |
| **Desktop Client** | Local GUI | PySide6 (Qt6), Open3D, PyTorch CUDA 12.8, Rasterio | Linux / Windows 11, NVIDIA RTX 30/40/50 Series GPU |

---

## 2. Central Server Deployment (Linux)

### Method A: Docker Compose Deployment (Recommended)
```bash
# 1. Clone your private server repository on the VPS
git clone https://github.com/bkabelik/lidar-server.git /opt/lidar-server
cd /opt/lidar-server

# 2. Configure production environment
cp .env.example .env
nano .env  # Adjust SECRET_KEY, DOMAIN, and POSTGRES_PASSWORD

# 3. Launch container stack
docker compose up -d --build

# 4. Verify running containers
docker compose ps
docker compose logs -f app
```

### Method B: Native Systemd Service (Debian / Ubuntu)
```bash
# 1. Install system prerequisites
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip postgresql nginx

# 2. Setup virtual environment
cd /opt/lidar-server
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Seed Root Sysadmin User
python cli.py create-admin --email admin@lidar.local --password "Admin1234!" --company "Kabelik GmbH"

# 4. Create Systemd Service: /etc/systemd/system/lidar-server.service
sudo tee /etc/systemd/system/lidar-server.service > /dev/null <<EOF
[Unit]
Description=LiDAR Cloud Central Service
After=network.target postgresql.service

[Service]
User=www-data
WorkingDirectory=/opt/lidar-server
ExecStart=/opt/lidar-server/venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now lidar-server
```

---

## 3. Storage VPS Agent Setup (13€ / 1 TB Nodes)

Instead of expensive cloud volumes, you can attach any number of 1 TB VPS instances (Hetzner, Netcup, Contabo @ ~13€/mo). Each storage node runs an ultra-lightweight agent (< 25 MB RAM) with kernel-level Nginx streaming.

### One-Click Turnkey VPS Installation:
```bash
# On your freshly installed 1 TB Storage VPS (Ubuntu/Debian):
# 1. Copy the storage_agent package from your repository to the VPS
scp -r /opt/lidar-server/storage_agent root@<STORAGE_VPS_IP>:/root/

# 2. SSH into the storage VPS and execute the installer
ssh root@<STORAGE_VPS_IP>
cd /root/storage_agent
sudo bash install.sh
```

**What `install.sh` does automatically:**
- Configures `/var/www/lidar-storage/deliverables` with byte-range streaming enabled.
- Deploys an optimized Nginx configuration with `sendfile on`, `tcp_nopush on`, and CORS enabled for browser 3D point cloud streaming.
- Installs a lightweight Python FastAPI agent running on port `9000` to receive chunked uploads directly from clients.
- Sets up automatic heartbeats to notify the central server of available free disk space.

### Connecting the Storage VPS to Central Admin:
1. Log into the Central Web Admin Panel at `https://<CENTRAL_DOMAIN>/admin`.
2. Navigate to the **Multi-VPS 1 TB Storage Pool** section.
3. Click **+ Add 1 TB Storage VPS**.
4. Enter the node label (e.g. `Hetzner-Frankfurt-01`) and public URL (e.g. `http://vps1.yourcompany.com:9000`).
5. The central server will immediately begin load-balancing new project uploads to the node with the most free capacity.

---

## 4. Linux Desktop Client Installation (LiDAR Workbench)

On Linux, LiDAR Workbench runs 100% unrestricted and open-source (unlimited projects and zero seat limits).

### 1. GPU Drivers & CUDA (Ubuntu / Debian / Pop!_OS / Fedora):
```bash
# Verify NVIDIA GPU is detected
nvidia-smi

# If drivers are needed on Ubuntu/Debian:
sudo apt install -y nvidia-driver-550
```

### 2. Python Environment Setup:
```bash
# 1. Clone client repository
git clone https://github.com/bkabelik/lidar-workbench.git ~/lidar-workbench
cd ~/lidar-workbench

# 2. Setup Conda environment (recommended for PyG and C++ extensions)
conda create -n lidar-workbench python=3.10 -y
conda activate lidar-workbench

# 3. Install PyTorch with matching CUDA 12.8 wheels
pip install torch==2.7.0+cu128 torchvision==0.22.0+cu128 --index-url https://download.pytorch.org/whl/cu128
pip install torch_scatter torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu128.html

# 4. Install desktop workbench requirements
pip install -r requirements.txt
pip install pyproj requests matplotlib

# 5. Place model checkpoint
mkdir -p models
# Place model_best.pth in ~/lidar-workbench/models/
```

### 3. Launching LiDAR Workbench:
```bash
conda activate lidar-workbench
cd ~/lidar-workbench
python main.py
```

### 4. Optional Desktop Launcher (`.desktop` file):
Create `~/.local/share/applications/lidar-workbench.desktop`:
```ini
[Desktop Entry]
Name=LiDAR Workbench
Comment=Professional Airborne & Drone LiDAR Processing
Exec=/home/YOUR_USER/miniconda3/envs/lidar-workbench/bin/python /home/YOUR_USER/lidar-workbench/main.py
Icon=/home/YOUR_USER/lidar-workbench/gui/assets/logo.png
Terminal=false
Type=Application
Categories=Graphics;Science;Geoscience;
```

---

## 5. Distributed Computing & Background Worker

**Zero CLI requirement on client workstations:**
- **Automatic Enrollment:** Logging in under *Help → License & Plan* automatically starts the background `CompanyNodeWorker` and registers this workstation in the company cluster with its GPU model and status `idle`.
- **Interactive Status Bar:** The bottom-right status pill shows live status: `🟢 Worker: Active (Idle)`, `🟡 Worker: Processing 'tile_001'`, or `⚪ Worker: Paused`.
- **Pause / Resume Control:** Clicking the status pill opens the *Company Compute Nodes* dialog. Click **Pause Worker** to temporarily free 100% GPU power for local interactive 3D rendering.
- **Batch Task Distribution:** In the *Pointcept Classification* dialog, switch to *"Distribute Across Company Nodes"*. Select online nodes and click *Start*. Workstations claim tiles round-robin and compute in parallel.

---

## 6. QGIS Integration & 3D Web Viewer

- **QGIS Symbology Extraction:** Under *Cloud → Publish Deliverables*, select your `.qgz` or `.qgs` project. Layer styles (boundary stroke colors, line widths, polygon fills) are extracted and converted to web-ready WGS84 GeoJSON.
- **Direct Storage Node Upload:** COPC point clouds and WebP orthophotos stream directly to the allocated 1 TB Storage VPS.
- **Open-Source MapLibre 3D Globe:** Generates a shareable URL (`/viewer/{share_token}`) using MapLibre GL + Deck.gl. Zero paid Mapbox or Cesium tokens required. 3D buildings are **OFF by default** for fast mobile/desktop load times.
- **Client Deliverable Downloads:** End-clients can download raw deliverables directly through the web viewer.

---

## 7. User Accounts & Security Verification

- **Zero Hardcoded Passwords:** No credentials or secrets exist in source code. All passwords use bcrypt hashing.
- **Initial Seed Admin:** `admin@lidar.local` / `Admin1234!` (Change immediately in `/admin`).
- **Company SSO & Seat Management:** Seats and project storage quotas are enforced entirely on the closed-source server.
- **Client Token Storage:** Desktop clients store only a short-lived Bearer JWT token in `~/.config/LiDAR_Workbench/license.json` (`%APPDATA%` on Windows).

---

## 8. Maintenance & Troubleshooting

- **Database Backup:**
  ```bash
  pg_dump -U lidar_user lidar_db > /backup/lidar_$(date +%Y%m%d).sql
  ```
- **Linux CUDA Verification:**
  ```bash
  python -c 'import torch; assert torch.cuda.is_available(); print("CUDA OK")'
  ```
- **Storage VPS Disk Health:** Central server automatically halts routing to nodes exceeding 90% capacity and shifts to the next available VPS in the pool.
- **Missing Open3D GL Libraries on Linux:**
  ```bash
  sudo apt install -y libgl1-mesa-glx libgomp1
  ```
