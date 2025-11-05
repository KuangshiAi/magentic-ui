#docker build --platform=linux/amd64 -t paraview-novnc .

docker run --platform=linux/amd64 -it --rm \
  -p 6080:6080 -p 5900:5900 -p 11111:11111 \
  -v ../data:/home/MCPagent/data \
  paraview-novnc