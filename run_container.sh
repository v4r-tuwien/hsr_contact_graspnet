# args not used atm
DOCKER_ARGS+=("-e DISPLAY=$DISPLAY")
DOCKER_ARGS+=("-v /tmp/.X11-unix:/tmp/.X11-unix")

xhost +local:docker

if [ "$(docker ps -aq --filter name=contact_graspnet_tf22)" ]; then
    print_info "Detected running container instance. Attaching to the running container"
    docker exec -it contact_graspnet_tf22 bash $@
    exit 0
fi

# Remove existing container instances to prevent conflicts when starting
if [ "$(docker ps -a --quiet --filter status=exited --filter name=ros_ml_container)" ]; then
    docker rm ros_ml_container > /dev/null
fi


docker build -t contact_graspnet_tf22 .
docker run --gpus all --network=host --env="DISPLAY" --env="QT_X11_NO_MITSHM=1"  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name=contact_graspnet_tf22 -t -d -v /home/stefan/contact_graspnet:/contact_graspnet contact_graspnet_tf22
#docker run --gpus all --network=host --name=ros_tf -t -d ros_tf

docker exec -it contact_graspnet_tf22 bash
