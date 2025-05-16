# SyncNet model
mkdir weights
mkdir weights/syncnet
wget http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model -O weights/syncnet/syncnet_v2.model
wget https://www.robots.ox.ac.uk/~vgg/software/lipsync/data/sfd_face.pth -O weights/syncnet/sfd_face.pth
# emonet
mkdir weights/emonet
wget https://github.com/face-analysis/emonet/raw/master/pretrained/emonet_8.pth -O weights/emonet/emonet_8.pth
wget https://github.com/face-analysis/emonet/raw/master/pretrained/emonet_5.pth -O weights/emonet/emonet_5.pth
