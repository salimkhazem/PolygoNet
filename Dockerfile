# Using Debian Bullseye as the base image
FROM debian:bullseye

# Set non-interactive installation mode
ENV DEBIAN_FRONTEND=noninteractive

# Update packages and install the necessary packages
RUN apt update 
RUN apt install -y ca-certificates gpg
RUN apt install -y build-essential cmake git 

RUN apt -y install g++
RUN apt -y install mesa-common-dev libglm-dev mesa-utils

RUN apt -y install libboost-all-dev

RUN apt -y install clang-9
RUN apt -y install libcgal*

RUN apt -y install libmagick++-dev
RUN apt -y install graphicsmagick*

RUN apt -y install libgmp-dev
RUN apt -y install libeigen3-dev
RUN apt -y install libfftw3-dev

RUN git clone https://github.com/DGtal-team/DGtal.git
RUN cd DGtal ; mkdir build ; cd build; cmake ..; make install

RUN git clone https://github.com/salimkhazem/MATC.git
RUN cd MATC/ImaGene; mkdir build; cd build; cmake ..; make -j
RUN cd MATC; mkdir build ; cd build ; cmake .. -DDGtal_DIR=../DGtal/build; make MATCSimplifiedDominantPoint -j
WORKDIR /MATC/src
COPY run.sh ./
RUN chmod +x ./run.sh

CMD ["./run.sh", "/path/to/input", "/path/to/output"]
#ENTRYPOINT ["MATC/build/MATCSimplifiedDominantPoint"]
