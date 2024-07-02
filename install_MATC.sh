apt install -y ca-certificates gpg

apt install -y build-essential cmake git


apt -y install g++
apt -y install mesa-common-dev libglm-dev mesa-utils

apt -y  install  libboost-all-dev

###Install clang-9
apt -y install clang-9
apt -y install libcgal*

apt -y install libmagick++-dev
apt -y install graphicsmagick*

# RUN apt -y install libinsighttoolkit4-dev
# RUN apt -y install doxygen
# RUN apt -y install libqglviewer-dev-qt5

apt -y install libgmp-dev
apt -y install libeigen3-dev
apt -y install libfftw3-dev

# DGTAL
git clone https://github.com/DGtal-team/DGtal.git
cd DGtal ; mkdir build ; cd build; cmake ..; make install 

# MATC 
cd ../../ 
git clone https://github.com/salimkhazem/MATC.git 
rm MATC/src/MATCSimplifiedDominantPoint MATC/src/ATCSimplifiedDominantPoint 
cd MATC/ImaGene; mkdir build; cd build; cmake ..; make -j 
cd ../../../  
cd MATC; mkdir build ; cd build ; cmake .. -DDGtal_DIR=../DGtal/build; make MATCSimplifiedDominantPoint -j 
mv MATCSimplifiedDominantPoint ../src/ 
