#FROM tensorflow/tensorflow:2.10.1-gpu
FROM tensorflow/tensorflow:2.3.4-gpu
FROM tensorflow/tensorflow:2.2.0-gpu # try at some point

MAINTAINER Stefan Thalhammer (thalhamm@technikum-wien.at)
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
RUN python -m pip install absl-py==1.0.0 #tensorflow 2.10.1 requires absl-py>=1.0.0
RUN python -m pip install astor==0.8.1
RUN python -m pip install cachetools==4.1.1 # cachetools 5.2.0 base
RUN python -m pip install decorator==4.4.2
RUN python -m pip install freetype-py==2.2.0
RUN python -m pip install gast==0.3.3 # gast 0.4.0 in base
RUN python -m pip install google-auth==1.23.0 # 2.14.1 in base
RUN python -m pip install google-auth-oauthlib==0.4.2 # 0.4.6 in base
RUN python -m pip install grpcio==1.33.2 # grpcio 1.50.0 in base
RUN python -m pip install idna==2.10 # idna 2.9 in base
RUN python -m pip install imageio==2.9.0
RUN python -m pip install importlib-metadata==4.4.0 # importlib-metadata=5.0.0 in base
RUN python -m pip install keras-applications==1.0.8 
RUN python -m pip install markdown==3.3.3 # Markdown 3.4.1 in base
RUN python -m pip install vtk==8.1.2
RUN python -m pip install mayavi==4.7.1
RUN python -m pip install networkx==2.5 
RUN python -m pip install oauthlib==3.1.0 # oauthlib 3.2.2 in base
RUN python -m pip install opencv-python #==4.4.0.46, newest, otherwise python module loading error 
RUN python -m pip install protobuf==3.13.0 # protobuf 3.19.6 in base
RUN python -m pip install pyglet==1.5.9 
RUN python -m pip install pyrender==0.1.43
RUN python -m pip install requests==2.24.0 # request 2.22.0 in base
RUN python -m pip install requests-oauthlib==1.3.0 # request-oauthlib 1.3.1 in base
RUN python -m pip install rsa==4.6 # rsa 4.9 in base
RUN python -m pip install rtree
#RUN python -m pip install tensorboard==2.10.1 # required by tf base
RUN python -m pip install tensorboard-plugin-wit==1.7.0 # 1.8.1 in base
#RUN python -m pip install tensorflow-estimator==2.10.0 # required by tf base
#RUN python -m pip install tensorflow-gpu==2.2.0 # see base image
RUN python -m pip install termcolor==1.1.0 # 2.1.0 in base
RUN python -m pip install urllib3==1.25.11 # 1.25.8 in base
RUN python -m pip install werkzeug==1.0.1 # 2.2.2 in base
RUN python -m pip install wrapt==1.12.1 # 1.14.1 in base
RUN python -m pip install zipp==3.4.0 # 3.10.0 in base
RUN python -m pip install python-fcl==0.0.12
RUN python -m pip install pyyaml==5.4
RUN python -m pip install -U tensorflow-gpu==2.2.0
RUN python -m pip install matplotlib
RUN python -m pip install PyQt5 
RUN python -m pip install -U rosdep
#RUN python -m pip install https://www.vtk.org/files/release/9.0/vtk-9.0.0-cp38-cp38-linux_x86_64.whl # somehow works with that version


RUN rosdep init && \
     rosdep update

RUN apt-get update && apt-get install --no-install-recommends -y --allow-unauthenticated \
     python3-catkin-tools \
     python3-pip \
     python3-all-dev \
     python3-catkin-pkg-module \
     python3-rospkg-modules \
     python3-empy \
     ros-melodic-desktop-full --fix-missing \
     ros-melodic-realsense2-camera \
     && rm -rf /var/lib/apt/lists/*

# tf2_ros import error: https://answers.ros.org/question/326226/importerror-dynamic-module-does-not-define-module-export-function-pyinit__tf2/

RUN python -m pip install --upgrade setuptools opencv-contrib-python

RUN echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
RUN mkdir -p ~/catkin_ws/src
RUN /bin/bash -c "source /opt/ros/melodic/setup.bash \
    && cd ~/catkin_ws \
    && catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3"
RUN echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
