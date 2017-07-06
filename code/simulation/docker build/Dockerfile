#
# Dockerfile to build the image host in docker hub: romarcg/traversability-ros-ubuntu-gazebo
#
FROM  ros:kinetic
RUN apt-get -y update && apt-get install -y ros-kinetic-desktop-full
CMD source /opt/ros/kinetic/setup.bash
##
# Core packages for ros+gazebo
##
RUN apt-get -y update && apt-get install -y ros-kinetic-ros-control libarmadillo-dev curl git python-catkin-tools dbus python-pip gfortran nano packagekit-gtk3-module libcanberra-gtk-module 
RUN apt-get -y update && apt-get install -y ros-kinetic-gazebo-ros-pkgs ros-kinetic-gazebo-ros-control ros-kinetic-rqt-common-plugins ros-kinetic-dynamic-reconfigure
##
# python packages
##
RUN /bin/bash -c " cd ~/ ; pip install scipy ; pip install numpy ; pip install sklearn; pip install pandas"
RUN apt-get -y update && apt-get install -y gedit eog nautilus
##
# workspace setup
##
RUN  /bin/bash -c "mkdir -p ~/catkin_ws/src ; source /opt/ros/kinetic/setup.bash ; cd ~/catkin_ws/src ; source /opt/ros/kinetic/setup.bash ; catkin_init_workspace; mkdir -p ~/dataset_cvs"
COPY src /root/catkin_ws/src
##
# For rviz or other software (e.g. gazebo) that need the use of xserver
##
LABEL com.nvidia.volumes.needed="nvidia_driver"
ENV PATH /usr/local/nvidia/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}
##
# Final command
##
CMD bash

##
# Executing with graphics and gpu enabled
##
# sudo -b nohup nvidia-docker-plugin > /tmp/nvidia-docker.log
#
# xhost  +
#
# nvidia-docker run -it --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v /home/omar/Codes/traversability_docker/traversability-ros-ubuntu-gazebo/volume:/volume --name="traversability_gazebo_task" traversability-ros-ubuntu-gazebo
