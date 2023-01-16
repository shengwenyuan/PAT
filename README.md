We use Swin Transformer code as the backbone,  which Microsoft have their copyright.  In the "models" folder, we build the code to support Patterns based Transformer.


environment setup

imagenet dataset format
imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...


pip install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2
pip install timm==0.4.12    
pip install opencv-python==4.4.0.46 termcolor==1.1.0 yacs==0.1.8  

train script 
python3.8 -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/pat/pat_tiny_patch4_224.yaml --data-path imagenet_path --batch-size 128 --accumulation-steps 8

throughput test script
python3.8 -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/pat/pat_tiny_patch4_224.yaml --data-path imagenet_path --batch-size 128 --throughput

