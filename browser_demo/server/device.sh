sudo apt update
sudo apt install -y linux-modules-extra-$(uname -r) v4l2loopback-dkms alsa-utils

# Load snd-aloop
sudo modprobe snd-aloop

# Load v4l2loopback with custom config
sudo modprobe v4l2loopback devices=1 video_nr=10 card_label="VirtualCam" exclusive_caps=1
