To configure the camera node OS to execute, monitor, and manage the script, perform the following on each node:
1. Download all required modules
    a. python -m pip install systemd asyncio opencv2-python threading numpy pyapriltags networktables --break
2. Copy robot_tracker.py to /home/camera_node/
3. Create a service file at /etc/systemd/system/camera_node.service with root permissions; i.e. sudo nano camera_node.service
4. Add the following configuration details to camera_node.service:
    [Unit]
    Description=Camera Node Script with Watchdog
    After=network.target

    [Service]
    Type=notify
    ExecStart=/usr/bin/python3 /home/camera_node/robot_tracker.py
    Restart=always
    RestartSec=0s
    WatchdogSec=2
    # Optional performance optimizations
    CPUSchedulingPolicy=rr  # Gives the process Real-Time round-robin priority
    CPUSchedulingPriority=99 # Maximum real-time priority (use with caution)

    [Install]
    WantedBy=multi-user.target
4. Install and configure the local time sync service
    a. sudo apt install -y chrony
    b. sudo systemctl stop systemd-timesyncd
    c. sudo systemctl disable systemd-timesyncd
    d. sudo nano /etc/chrony/chrony.conf
    e. Add the following line to the top of the file and then save it (CTRL + X, then Y, then ENTER)
        server XXX.XXX.XXX.XXX iburst
    f. sudo systemctl restart chrony
    g. chronyc sources -v
    h. Verify that its working, by seeing the asterisk next to the device IP address
    i. sudo chronyc makestep
5. Enable and start the service:
    a. Run sudo systemctl daemon-reload to read the new file.
    b. Run sudo systemctl enable camera_node.service to start it at boot.
    c. Run sudo systemctl start camera_node.service to run it right now.