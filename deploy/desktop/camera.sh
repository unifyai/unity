# Set up for virtual camera (host.sh)
sudo apt-get update && sudo apt-get install -y v4l2loopback-dkms v4l2loopback-utils v4l-utils linux-modules-extra-$(uname -r)

sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="TestCam" exclusive_caps=1

sudo usermod -aG video $USER
newgrp video
