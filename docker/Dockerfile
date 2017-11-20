FROM ubuntu:xenial
RUN apt-get update

# I'm assuming the following packages are already installed in
# the WT container
RUN apt-get -y install curl python3-pip git
CMD ["/bin/bash"]
WORKDIR /tmp
VOLUME ["/sils"]
