import os
import torch
import numpy as np
import h5py
from scipy import ndimage

import torch.nn as nn
import torchvision.models.video as models
import torch
from torch.utils.data import Dataset, DataLoader

import torch.optim as optim
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
import pandas as pd
import csv
import random

import matplotlib.pyplot as plt


def load_file_paths():
    # Get the directory where this script is located
    base_dir = os.path.dirname(__file__)
    
    # Path to the "samples" subfolder
    samples_dir = os.path.join(base_dir, "samples")
    
    # List to hold full file paths
    filePaths = []
    classLabel=[]
    
    # Walk through the folder and collect .h5 files
    for root, _, files in os.walk(samples_dir):
        for file in files:
            if file.endswith(".h5"):
                full_path = os.path.join(root, file)
                label=1 #here 1 means cancer samples
                filePaths.append(full_path)
                classLabel.append(label)
                 
    NumberOfDrugSamples= len(filePaths) 
    print("Number of Drug Screening Samples", NumberOfDrugSamples)    
    
    return np.array(filePaths), np.array(classLabel)

def z_score_normalize(image):
    mean = np.mean(image)
    std = np.std(image)
    normalized_image = (image - mean) / std
    return normalized_image

def normalize(volume):
    """Normalize the volume"""
    min = 0
    max = 20000
    volume[volume < min] = min
    volume[volume > max] = max
    volume = (volume - min) / (max - min)
    volume = volume.astype("float32")
    return volume

def resize_volume(img):
    
    # Set the desired depth
    desired_x = 131
    desired_y = 140
    desired_z = 228
    
    current_x = img.shape[0]
    current_y = img.shape[1]
    current_z = img.shape[2]
    
    # Compute factor
    
    width = current_x / desired_x
    height = current_y / desired_y
    depth = current_z / desired_z
    
    width_factor = 1 / width
    height_factor = 1 / height
    depth_factor = 1 / depth

    img = ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
    return img

def process_scan(volume):
    """Normalize and resize volume"""
    # Normalize
    #volume = normalize(volume)
    volume=z_score_normalize(volume)
    # Resize width, height and depth
    volume = resize_volume(volume)
    return volume


def load_and_preprocess_3d(file_path):
    tempData=h5py.File( file_path, 'r')
    rawData=tempData['dataset_1'][:,:,:]
    rawData=rawData.astype(np.float32)
    #rawData_reshape_like_matlab=np.transpose(rawData[:,:,:], (2,1,0))
    rawData_reshape_like_matlab=rawData[:,:,:]
    #print(rawData_reshape_like_matlab.shape)
    rawData=process_scan(rawData_reshape_like_matlab)
    return rawData

class My3DDataset(Dataset):
    def __init__(self,  filePath_train_test, y_train_val_test, transform=None):
        self.sample_y =y_train_val_test
        self.transform = transform
        self.samples = filePath_train_test
        
        #print(self.sample_y[0])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        #print(sample_path)
        #print(idx)
        label=self.sample_y[idx]
        #print(label)
        sample = load_and_preprocess_3d(sample_path)
        sample = np.stack((sample,) * 3, axis=0)
        label=label.astype(np.int64)
        return sample, label
    
def voting_for_drug_screening(ensemble_all_predictions, ensemble_all_probs, classifier_weights, weighted_mean_flag):
    
    voting_result_all_predictions=[]
    voting_result_all_probs=[]
    mean_of_all_probs=[]
    
    for s_num in  range(len(ensemble_all_probs[0])):
    #for s_num in  range(2):
        num_positive=0
        num_negative=0
        num_pos_prob=0
        num_neg_prob=0
        total_probs=0
        total_classifier=len(ensemble_all_predictions)
        
        total_weights=0
        
        for m_num in range(total_classifier): 
            
            if weighted_mean_flag==1:
                total_probs=total_probs+float(classifier_weights[m_num])*ensemble_all_probs[m_num][s_num]
                total_weights=total_weights+classifier_weights[m_num]
                #print("Weighted Mean is calculating")
            else:
                total_probs=total_probs+ensemble_all_probs[m_num][s_num]
                #print("Normal Mean is calculating")
                total_weights=total_classifier
                
            if ensemble_all_predictions[m_num][s_num]==1:
                num_positive=num_positive+1
                num_pos_prob=num_pos_prob+ensemble_all_probs[m_num][s_num]
            else:
                num_negative=num_negative+1
                num_neg_prob=num_neg_prob+(1-ensemble_all_probs[m_num][s_num])
                
        if(num_positive>num_negative):
            voting_result_all_predictions.append(1)
            avg_prob=num_pos_prob/num_positive
            voting_result_all_probs.append(avg_prob)
            
        if(num_negative>num_positive):
            voting_result_all_predictions.append(0)
            avg_prob=num_neg_prob/num_negative
            voting_result_all_probs.append(avg_prob)   
        
        if(num_positive==num_negative):
            if(num_pos_prob>num_neg_prob):
                voting_result_all_predictions.append(1)
                avg_prob=num_pos_prob/num_positive
                voting_result_all_probs.append(avg_prob) 
            if(num_neg_prob>num_pos_prob):
                voting_result_all_predictions.append(0)
                avg_prob=num_neg_prob/num_negative
                voting_result_all_probs.append(avg_prob)    
            if(num_pos_prob==num_neg_prob):
                print("Rare Case")
                toss=random.randint(0, 1)
                if(toss==1):
                    voting_result_all_predictions.append(1)
                    avg_prob=num_pos_prob/num_positive
                    voting_result_all_probs.append(avg_prob) 
                else:
                    voting_result_all_predictions.append(0)
                    avg_prob=num_neg_prob/num_negative
                    voting_result_all_probs.append(avg_prob) 
        
        mean_probability=total_probs/total_weights
        mean_of_all_probs.append(mean_probability) 
                         
    #print(num_positive)   
    #print(num_negative)   
    #print(num_pos_prob) 
    #print(num_neg_prob)      
    
    return voting_result_all_predictions, voting_result_all_probs, mean_of_all_probs



# Function to save checkpoint
def save_checkpoint(state, filename):
    torch.save(state, filename)
    

if __name__ == "__main__":
    
    filePaths, classLabel=load_file_paths()
    
    best_model_date = np.array([
    "20231114", "20231120", "20231121", "20231122", "20231123", "20231127", "20231128",
    "20231129", "20240224", "20240225", "20240731", "20240813", "20240816", "20240817",
    "20240819", "20240826", "20240827", "20240828", "20240829", "20240902", "20240904",
    "20241003", "20241017"
    ])

    classifier_weights = np.array([
    0.961538462, 0.935344828, 0.945544554, 0.966527197, 0.984, 0.893333333, 0.819796954,
    0.893491124, 0.887323944, 0.902777778, 0.96875, 0.907692308, 0.83974359, 0.857142857,
    0.867768595, 0.959459459, 0.957894737, 0.931677019, 0.867924528, 0.956521739,
    0.918367347, 0.831521739, 0.981595092
    ])

    X_test_file_path=filePaths
    y_test=classLabel
    number_class=2
    
    print("Test Samples Size")
    print(X_test_file_path.shape)
    

    test_data_load_from_disk = My3DDataset(X_test_file_path, y_test)
    test_dataset = DataLoader(test_data_load_from_disk, batch_size=2, shuffle=False)
    
    ensemble_all_labels=[]
    ensemble_all_predictions=[]
    ensemble_all_probs=[]
    
    
    for j in range(0, best_model_date.shape[0]):
    #for j in range(0, 1):
        
        # Load the pre-trained 3D CNN model
        model = models.mc3_18(pretrained=True)

        # Modify the final layer to fit binary classification (output a single logit)
        model.fc = nn.Linear(model.fc.in_features, number_class)

        # Move the model to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        # Get the directory where this script is located
        base_dir = os.path.dirname(__file__)
        
        # Path to the "samples" subfolder
        model_dir = os.path.join(base_dir, "models")

        file_name_for_best_model=model_dir+"/best_model_for_date_" + best_model_date[j] + ".tar"
        # Load the best model
        best_model = torch.load(file_name_for_best_model)
        model.load_state_dict(best_model['state_dict'])

        # Evaluate on test set
        print('Started Testing using the best model of ...', j, best_model_date[j])
        model.eval()
        all_labels = []
        all_predictions = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in test_dataset:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                probs = nn.functional.softmax(outputs, dim=1)
                #print(outputs.cpu().numpy())
                #print(probs.cpu().numpy())
                
                _, predicted = torch.max(outputs, 1)
                #print(predicted.cpu().numpy())
                #print(probs.cpu().numpy()[:, 1])
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                all_probs.extend(probs.cpu().numpy()[:, 1])
                
        all_labels = np.array(all_labels)
        all_predictions = np.array(all_predictions)
        all_probs = np.array(all_probs)
        
        ensemble_all_labels.append(all_labels)
        ensemble_all_predictions.append(all_predictions)
        ensemble_all_probs.append(all_probs)
    
    voting_result_all_predictions, voting_result_all_probs, mean_of_probs=voting_for_drug_screening(ensemble_all_predictions, ensemble_all_probs, classifier_weights, 1)    
    voting_result_all_predictions=np.array(voting_result_all_predictions)
    voting_result_all_probs=np.array(voting_result_all_probs)
    mean_of_probs=np.array(mean_of_probs)
    true_all_labels=ensemble_all_labels[0]
    
    file_name_for_test= "test_result.csv"
    df_test_data = pd.DataFrame({"File Name" : X_test_file_path, "Probability of clossness to Cancer Morphology":mean_of_probs})
    df_test_data.to_csv(file_name_for_test, index=False)
    print("Test result has been prodoced and save to test_result.csv. Please check that file.")