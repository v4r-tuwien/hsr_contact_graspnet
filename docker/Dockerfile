#FROM tensorflow/tensorflow:2.10.1-gpu
FROM tensorflow/tensorflow:2.3.4-gpu
# FROM tensorflow/tensorflow:2.2.0-gpu # try at some point

ENV DEBIAN_FRONTEND noninteractive
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
      nvidia-driver-470 \
      && rm -rf /var/lib/apt/lists/*


RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

#RUN apt-get update && apt-get install curl

RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
      ros-melodic-desktop-full \
      python3-tqdm \
      vim \
      git \
      && rm -rf /var/lib/apt/lists/*

#RUN echo "export QT_X11_NO_MITSHM=1" >> ~/.bashrc

# modified contact-graspnet requirements
COPY requirements.txt /root/requirements.txt
RUN python -m pip install --upgrade pip setuptools wheel
RUN python -m pip install -r /root/requirements.txt
RUN python -m pip install mayavi==4.7.1
RUN python -m pip install opencv-python-headless==4.3.0.38

RUN rosdep init && \
     rosdep update

RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-catkin-tools \
     python3-pip \
     python3-all-dev \
#      python3-catkin-pkg-module \
     python3-rospkg-modules \
     python3-empy \
     ros-melodic-desktop-full --fix-missing \
     ros-melodic-realsense2-camera \
     && rm -rf /var/lib/apt/lists/*

# tf2_ros import error: https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/

RUN python -m pip install \
    catkin_pkg \
    rospkg

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN mkdir -p ~/catkin_ws/src
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash"
#     && cd ~/catkin_ws \
#     && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3"

# Clone all ROS packages first
RUN git clone -b noetic-devel --depth 1 https://github.com/ros-perception/vision_msgs.git /root/catkin_ws/src/vision_msgs
RUN git clone -b melodic --depth 1 https://github.com/ros-perception/vision_opencv.git /root/catkin_ws/src/vision_opencv
RUN git clone https://github.com/v4r-tuwien/object_detector_msgs.git /root/catkin_ws/src/object_detector_msgs

# Now configure and build once with Python 3
RUN /bin/bash -c '. /opt/ros/melodic/setup.bash && \
    cd /root/catkin_ws && \
    catkin config -DPYTHON_EXECUTABLE=/usr/bin/python3 \
                  -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m \
                  -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so && \
    catkin build'

RUN python -m pip install rosnumpy

WORKDIR /root