# scale-aware-SSD

### Dataset
-------------------------
TomatoD is a task-specific highly specialized object detection and classification dataset of tomato fruits that apart from bounding box information, it also contains class information for three ripening stages of each tomato fruit.

### Download the data
-------------------------
* [Images](https://datasets-u2m.s3.eu-west-3.amazonaws.com/tomatOD_images.zip)
* [Bounding box annotations (train and validation sets)](https://datasets-u2m.s3.eu-west-3.amazonaws.com/tomatOD_annotations.zip)

### Data Format
-------------------------
The annotations of the tomatOD dataset are provided in a COCO compatible format.

### Statistics and data analysis
-------------------------
##### tomatOD classes
The table below shows the number of annotated objects for each class of the **tomatOD** dataset.

| unripe | semi-ripe | fully-ripe |
|:------:|:---------:|:----------:|
| 1592   | 395       | 431        |

### Dataset Extraction
-------------------------
```bash
	# Install zip
	apt-get install unzip
	
	# Unzip directories
	unzip tomatOD_images.zip
	unzip tomatOD_annotations.zip
	
	# Remove zips
	rm tomatOD_images.zip
	rm tomatOD_annotations.zip
	
	# Make dataset directory
	mkdir -p scaleSSD/storage/tod/
	
	# Copy images
	cp -r tomatOD_images/* scaleSSD/storage/tod/
	
	# Copy annotations
	cp tomatOD_annotations/tomatOD_train.json scaleSSD/storage/tod/train/
	cp tomatOD_annotations/tomatOD_test.json scaleSSD/storage/tod/test/
```

### Requirements
-------------------------

```bash
	apt update						
	apt install python3.8			
	apt install python3-pip			
	pip3 install 				\
		albumentations==0.5.2    	\
		matplotlib==3.4.1    		\
		numpy==1.20.2			\
		opencv-python==4.5.1.48 	\
		pycocotools==2.0.2    		\
		pytorch-lightning==1.2.6   	\
		tensorboard==2.4.1		\
		torch==1.8.1+cu111		\
		torchvision==0.9.1+cu111	\
		wheel==0.35.1			\
		scikit-learn==0.22.2		\
		scipy==1.6.2
```
**NVIDIA-SMI:**
460.39       
**Driver Version:**
460.39       
**CUDA Version:**
11.2
**Python:**
3.8.5

### Run:
-------------------------

**Training:**

```bash
	python3 -m scaleSSD.train \
		--train_images *train_images_dir* \
		--train_labels *train_labels_path* \
		--test_images *test_images_dir* \
		--test_labels *test_labels_path*
```
Example:
```bash
	python3 -m scaleSSD.train \
		--train_images scaleSSD/storage/tod/train/img \
		--train_labels scaleSSD/storage/tod/train/tomatOD_train.json \
		--test_images scaleSSD/storage/tod/test/img \
		--test_labels scaleSSD/storage/tod/test/tomatOD_test.json
```

**Evaluation:**

```bash
	python3 -m scaleSSD.eval \
		--train_images *train_images_dir* \
		--train_labels *train_labels_path* \
		--test_images *test_images_dir* \
		--test_labels *test_labels_path* \
		--ckpt_path *checkpoint_path*
```
Example:
```bash
	python3 -m scaleSSD.eval \
		--train_images scaleSSD/storage/tod/train/img \
		--train_labels scaleSSD/storage/tod/train/tomatOD_train.json \
		--test_images scaleSSD/storage/tod/test/img \
		--test_labels scaleSSD/storage/tod/test/tomatOD_test.json \
		--ckpt_path output/custom_ssd_ckpt/default/version_9/checkpoints/
```


### Training procedure:
-------------------------
Basic steps of the training procedure:

	1. Initialize hyper parameters
			1. data parameters
					1. batch size
					2. train, test paths
					3. input size
			2. learning parameters
					1. class weights
					2. minibatch size
					3. sampler choice ( dns, basic 3/1 ) 
					4. learning rate
			3. model parameters
					1. anchor dimensions
					2. number of classes 
			4. training parameters
					1. number of epochs
			5. output parameters
					1. output folder
	2. Create SSD model
			1. Generate Feauture Maps
			2. Initialize model
			3. Inialize bbox encoder
			4. Inialize bbox decoder
			5. Inialize bbox nms
	2. Initialize Loss Detection Object 
	3. Initialize Detection Model
	4. Initialize Trainer Object
	5. Initialize Data Module
	6. Start Training with Training set
	7. Test Model wih Test set
	8. Run Coco Evaluator to acquire metrics


### Build Docker Image
-------------------------
```bash
	docker build -t scale_aware_ssd:0.0.1 -f Dockerfile .
```
