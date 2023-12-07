# args not used atm
DOCKER_ARGS+=("-e DISPLAY=$DISPLAY")
DOCKER_ARGS+=("-v /tmp/.X11-unix:/tmp/.X11-unix")

xhost +local:docker

if [ "$(docker ps -aq --filter name=contact_graspnet_final)" ]; then
    print_info "Detected running container instance. Attaching to the running container"
    docker exec -it contact_graspnet_final bash $@
    exit 0
fi

# Remove existing container instances to prevent conflicts when starting
if [ "$(docker ps -a --quiet --filter status=exited --filter name=contact_graspnet_final)" ]; then
    docker rm ros_ml_container > /dev/null
fi

# need to set that if cam cannot be found?
# udevadm control --reload-rules && udevadm trigger

xhost +local:docker
#docker build -t contact_graspnet_tf22 .
docker run --gpus all --network=host --env="DISPLAY" --env="QT_X11_NO_MITSHM=1"  --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" --name=contact_graspnet_final -t -d -v /home/stefan/contact_graspnet:/contact_graspnet -v /dev:/dev --device-cgroup-rule "c 81:* rmw"  --device-cgroup-rule "c 189:* rmw" contact_graspnet_final
#docker run --gpus all --network=host --name=ros_tf -t -d ros_tf
xhost -local:root

docker exec -it contact_graspnet_final bash
