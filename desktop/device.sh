# Create the virtual sink/mic
pipewire &
pipewire-pulse &
wireplumber &
sleep 2

# 1. For capturing Meet participant audio
pactl load-module module-null-sink sink_name=meet_sink
pactl load-module module-remap-source master=meet_sink.monitor source_name=meet_mic

# 2. For agent TTS (only goes to Meet, not to agent itself)
pactl load-module module-null-sink sink_name=agent_sink
pactl load-module module-remap-source master=agent_sink.monitor source_name=agent_mic

pactl set-default-source meet_mic
pactl set-default-sink agent_sink
