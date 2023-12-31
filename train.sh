mkdir data/coco 
mkdir data/cocostuffthings
mkdir logs/coco_ckpt

# wget http://images.cocodataset.org/zips/train2017.zip
# wget http://images.cocodataset.org/zips/val2017.zip
# wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# wget http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip
# wget https://heibox.uni-heidelberg.de/f/78dea9589974474c97c1


# mv coco_epoch117.ckpt ./logs/coco_ckpt

# unzip -d ./data/coco train2017.zip
# unzip -d ./data/coco val2017.zip
# unzip -d ./data/coco annotations_trainval2017.zip
# unzip -d ./data/cocostuffthings stuffthingmaps_trainval2017.zip

python main.py --base configs/custom_vqgan.yaml -t True --gpus 0,1,2,3 -r ./logs/2023-*******全放开前cross_attention/checkpoints/last.ckpt