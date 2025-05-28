# Browser Use in Server

1. Build Dockerfile

`docker build -t bu_test -f Dockerfile .

2. Run Docker container

`docker run -p 6080:6080 -rm bu_test

Note: add `-d` for detached mode

3. Live view the container browser locally at `http://localhost:6080/vnc.html?autoconnect=true&resize=scale&reconnect=true`