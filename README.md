# Lung-Cancer-Prediction-Using-CNN-and-Transfer-Learning

# INTRODUCTION ---

Lung cancer is one of the leading causes of cancer-related deaths worldwide and its mortality rate is 19.4%. Early detection of lung tumor is done by using many imaging techniques such as Computed Tomography (CT), Sputum Cytology, Chest X-ray and Magnetic Resonance Imaging (MRI). Detection means classifying tumor two classes (i)non-cancerous tumor (benign) and (ii)cancerous tumor (malignant). 

Neural network plays a key role in the recognition of the cancer cells among the normal tissues, which in turn provides an effective tool for building an assistive AI based cancer detection. Accurate classification and early detection are essential for effective treatment and improving patient survival. 

The aim of the project is to develop an efficient and accurate model for predicting lung cancer from medical images, specifically CT scans, using Convolutional Neural Networks (CNN) and transfer learning.  

Lung cancer is a critical health challenge, and early detection is crucial for improving patient outcomes. Convolutional Neural Networks (CNNs) have proven to excel in image analysis tasks, and Transfer Learning allows for leveraging existing models to enhance performance and efficiency. The use of chest X-ray or CT scan images is standard for lung cancer diagnosis. The combination of CNN and Transfer Learning represents a modern, effective approach to developing advanced diagnostic tools for cancer detection. 

 The model classifies lung cancer images into four categories:  Normal, Adenocarcinoma Large Cell Carcinoma and Squamous Cell Carcinoma. This project utilizes deep learning methods to create a reliable model for classifying lung cancer using chest X-ray images. By using Convolutional Neural Networks (CNNs) and transfer learning, the project's approach can achieve higher accuracy in classifying lung cancer images compared to traditional methods.  

# MAIN CODE ---
https://drive.google.com/file/d/1C48bb1g_qNYI4t9puNeDqoEQviU0AKYE/view?usp=drive_link

# REQUIREMENTS ---
numpy==1.26.3

streamlit==1.30.0

tensorflow==2.15.0.post1

# HOW TO USE ---
1. Open the Application
Launch the website or application built using Streamlit for lung cancer prediction.

2. Navigate to the Home Page
The Home Page provides an overview of the application, including its purpose and how it works.
Review the information to understand the project.

3. Go to the Predict Page
Click on the "Predict" tab or button from the navigation menu to access the prediction functionality.

4. Upload a CT Scan Image
On the Predict Page, locate the image upload section.
Click the "Browse Files" button to upload a CT scan image from your device.
Ensure the uploaded image is in the required format (e.g., .png or .jpg).

5. Start the Prediction
After uploading the image, the system automatically preprocesses it and feeds it into the trained deep learning model.
Wait for the prediction results to be displayed.

6. View the Prediction Results
The interface will show:
The predicted type of lung cancer (e.g., Adenocarcinoma, Large Cell Carcinoma, Squamous Cell Carcinoma, or Normal).
The probability scores for each category.

7. Explore the About Section
Navigate to the "About" tab to learn more about the project, including the underlying technology, datasets, and objectives.

8. Exit the Application
Close the browser or application after reviewing the predictions.

# RESULT ---
The model achieved a classification accuracy of 93%, demonstrating its ability to accurately predict lung cancer types based on CT scan images. The results are compared with traditional diagnostic methods, showing significant improvement in accuracy and diagnostic time. The model’s 93% classification accuracy reflects its robust capability to analyse and predict lung cancer types from CT scan images, a critical advancement in oncological diagnostics. When compared to traditional methods, which often rely on manual interpretation by radiologists, this model not only improves accuracy but also significantly reduces diagnostic time. Traditional approaches can take hours or even days to yield results, while the CNN-based model can provide near-instantaneous predictions, facilitating quicker treatment decisions.  
  

# OUTPUT ---
![1ss](https://github.com/user-attachments/assets/010be3a3-fe6d-4bd7-b2d6-4e81802b1a1d)
![2ss](https://github.com/user-attachments/assets/c04601b2-95ef-43ab-a2b2-fce6170cdcaa)
![3ss](https://github.com/user-attachments/assets/3f600b81-a324-464a-a69b-393eeff496c2)
![4ss](https://github.com/user-attachments/assets/f09fe632-d6a0-414b-8427-50e37e1f3591)
![5ss](https://github.com/user-attachments/assets/4404f059-96e0-4f2d-a224-9d24dc682bbf)





