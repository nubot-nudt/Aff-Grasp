# Stable-Grasp-Generation-Enabled-by-Affordance-Understanding

## 🚀 Project Progress  

- [x] Open-source the dataset  
  Dataset is published at [this link](https://drive.google.com/file/d/1FEhQ0zIeFJEHcWFkZEcfL5WFIMX9jI6T/view?usp=drive_link)  with raw data.
  The code for building the dataset in Vrep or Sapien has provided.

  The annotation results of objects in GraspNet-1Billion can be downloaded at [this link](https://www.modelscope.cn/datasets/artguang/Semantic_annotation_of_models_in_GraspNet-1Billion/resolve/master/graspnet_annotation.zip).

- [ ] **Open-source code**  
  ETA: Soon

  

## 📋 Pending Tasks  

- [ ] Create                  core code documentation  

- [ ] Add usage examples and demos  

- [ ] Setup contribution guidelines  

  

## 📂 Dataset Structure  

├── Chair

&nbsp;&nbsp;├── object_0

&nbsp;&nbsp;&nbsp;&nbsp;├── sem

​&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;​├── train_0.png...train_199.png&nbsp;&nbsp;&nbsp;&nbsp;# train images with annotation

​&nbsp;&nbsp;&nbsp;&nbsp;├── train

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── train_0.png...train_199.png&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# train images 

&nbsp;&nbsp;&nbsp;&nbsp;├── transforms_train.json&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# poses of train images 

├── Dispenser

├── Door

├── Faucet

├── Kettle

├── Keyboard

├── Kitchenpot

├── Lamp

├── StorageFurniture

├── Table

└── Toilet
