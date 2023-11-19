if [ "$(docker ps -aq --filter name=contact_graspnet)" ]; then
    print_info "Detected running container instance. Attaching to the running container"
    docker exec -it contact_graspnet bash $@
    exit 0
fi

docker build -t contact_graspnet .
docker run --network=host --name=contact_graspnet -t -d -v /home/stefan/contact_graspnet:/contact_graspnet contact_graspnet
#docker run --gpus all --network=host --name=ros_tf -t -d ros_tf

docker exec -it contact_graspnet bash
