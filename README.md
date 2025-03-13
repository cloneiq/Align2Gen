# Align2Gen：A Lesion-Enhanced Dual-Branch Model for High-Quality Radiology Report Generation
 ## Overview
 
 Radiology report generation is valuable in assisting diagnosis, reducing doctors’ workload, and improving accuracy by automatically generating diagnostic reports by integrating radiological image content with clinical knowledge. However, most existing models primarily establish coarse-grained mappings between global images and texts, ignoring the fine-grained relationship between lesion regions and report content, which affects report accuracy. 
 
 To this end, this paper proposes Align2Gen, a radiology report generation model designed for lesion perception.  
 
 - we introduce semi-supervised learning to generate bounding box annotations for the MIMIC-CXR dataset through the Unbiased-Teacher v2 framework, which reduces the reliance on manual annotations and improves the efficiency and coverage of annotations.
 - we propose the Align2Gen model to improve the accuracy and interpretability of the report generation by combining the lesion area enhancement module (LERA) and bounding box annotation to focus more on clinically important lesion areas.
 - we design a global-local bi-branch implicit alignment module (LAB and GAB) to enhance feature alignment between vision and text and reduce information mismatch. 
 
 ## Usage
 ### Setup
 #### Align2Gen Setup
 ```
 python 3.8
 pytorch 1.10.0
 cuda 11.3
 ```
 ### Downloading necessary data
- You can access the official download page for the MIMIC-CXR dataset from the following [Link](https://physionet.org/content/mimic-cxr/2.0.0/).
- You can access the official download page for the VinDr-CXR dataset from the following [Link](https://physionet.org/content/vindr-cxr/1.0.0/).
- If you want to obtain the VinDr-CXR dataset in JPG format, you can download it from the Kaggle competition platform.
 ### Train and Test
 ```
 cd Align2Gen
 python main.py
 ```
 
 ##  License
 The source code is free for research and education use only. Any commercial use should get formal permission first.
 
 ## Acknowledgement
- Thanks [unbiased-teacher-v2](https://github.com/facebookresearch/unbiased-teacher-v2) for serving as building blocks of Align2Gen.
- Thanks [R2Gen](https://github.com/zhjohnchan/R2Gen) for serving as building blocks of Align2Gen.
