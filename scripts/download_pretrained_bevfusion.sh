mkdir bev_lib/pretrained &&
cd bev_lib/pretrained &&
wget https://bevfusion.mit.edu/files/pretrained/lidar-only-seg.pth -O lidar-only-seg.pth&&
wget https://bevfusion.mit.edu/files/pretrained/camera-only-seg.pth -O camera-only-seg.pth&&
wget https://bevfusion.mit.edu/files/pretrained/bevfusion-seg.pth -O bevfusion-seg.pth&&
cd ../../
