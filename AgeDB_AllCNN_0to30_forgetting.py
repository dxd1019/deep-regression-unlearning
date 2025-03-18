# ----------------------------------------
# IMPORT NECESSARY PACKAGES
# ----------------------------------------

import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from utils import *
from models import AllCNN
from datasets import AgeDB
from unlearn import *
from metrics import *
from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

import nltk
nltk.download('stopwords')



# ----------------------------------------
# LOAD DATASET AND PREPARE DATA LOADERS
# ----------------------------------------

# Load dataset from CSV file
print("Loading dataset...")
df = pd.read_csv("./data/agedb.csv")

# Split dataset into training, validation, and test sets
df_train, df_val, df_test = df[df['split'] == 'train'], df[df['split'] == 'val'], df[df['split'] == 'test']

# Initialize dataset objects
print("Initializing dataset objects...")
train_data = AgeDB(data_dir='./data', df=df_train, img_size=32, split='train')
val_data = AgeDB(data_dir='./data', df=df_val, img_size=32, split='val')
test_data = AgeDB(data_dir='./data', df=df_test, img_size=32, split='test')

# Create DataLoaders for training, validation, and testing
print("Creating DataLoaders...")
train_loader = DataLoader(train_data, batch_size=256, shuffle=True,
                              num_workers=64, pin_memory=True, drop_last=False)
val_loader = DataLoader(val_data, batch_size=256, shuffle=False,
                            num_workers=64, pin_memory=True, drop_last=False)
test_loader = DataLoader(test_data, batch_size=256, shuffle=False,
                             num_workers=64, pin_memory=True, drop_last=False)



# ----------------------------------------
# LOAD AND TRAIN THE MODEL
# ----------------------------------------

device = 'cuda'
print(f"Using device: {device}")

# Initialize the model
print("Initializing model...")
model = AllCNN(num_classes=1).to(device)
epochs = 100

# Load pre-trained model weights
save_path = "../saved_models/AllCNN_AgeDB_100epochs.pt"
#history = fit_one_cycle(epochs, model, train_loader, val_loader, device = device, save_path = save_path)

print(f"Loading model weights from {save_path}...")
model.load_state_dict(torch.load(save_path))



# ----------------------------------------
# CREATE FORGET AND RETAIN SETS
# ----------------------------------------

print("Creating forget and retain datasets...")

# Separate data based on age threshold (≤30 for forget set, >30 for retain set)
df_train_forget = df_train[df_train['age'] <= 30]
df_val_forget = df_val[df_val['age'] <= 30]
df_test_forget = df_test[df_test['age'] <= 30]

# Create dataset and DataLoaders for forget set
train_data_forget = AgeDB(data_dir='./data', df=df_train_forget, img_size=32, split='train')
val_data_forget = AgeDB(data_dir='./data', df=df_val_forget, img_size=32, split='val')
test_data_forget = AgeDB(data_dir='./data', df=df_test_forget, img_size=32, split='test')

train_forget_loader = DataLoader(train_data_forget, batch_size=256, shuffle=False,
                              num_workers=64, pin_memory=True, drop_last=False)
val_forget_loader = DataLoader(val_data_forget, batch_size=256, shuffle=False,
                            num_workers=64, pin_memory=True, drop_last=False)
test_forget_loader = DataLoader(test_data_forget, batch_size=256, shuffle=False,
                             num_workers=64, pin_memory=True, drop_last=False)

df_train_retain = df_train[df_train['age'] > 30]
df_val_retain = df_val[df_val['age'] > 30]
df_test_retain = df_test[df_test['age'] > 30]

# Create dataset and DataLoaders for retain set
train_data_retain = AgeDB(data_dir='./data', df=df_train_retain, img_size=32, split='train')
val_data_retain = AgeDB(data_dir='./data', df=df_val_retain, img_size=32, split='val')
test_data_retain = AgeDB(data_dir='./data', df=df_test_retain, img_size=32, split='test')

train_retain_loader = DataLoader(train_data_retain, batch_size=256, shuffle=True,
                              num_workers=64, pin_memory=True, drop_last=False)
val_retain_loader = DataLoader(val_data_retain, batch_size=256, shuffle=False,
                            num_workers=64, pin_memory=True, drop_last=False)
test_retain_loader = DataLoader(test_data_retain, batch_size=256, shuffle=False,
                             num_workers=64, pin_memory=True, drop_last=False)



# ----------------------------------------
# EVALUATE ORIGINAL MODEL
# ----------------------------------------

print("Evaluating the original model...")

evaluate(model, test_retain_loader, 'cuda')

evaluate(model, test_forget_loader, 'cuda')



# ----------------------------------------
# TRAIN A NEW MODEL ON RETAINED DATA
# ----------------------------------------

print("Training a retrained model on retain data...")
device = 'cuda'
gold_model = AllCNN(num_classes=1).to(device)
epochs = 100
save_path = "../saved_models/AllCNN_AgeDB_100epochs_0to30_Gold.pt"
#history = fit_one_cycle(epochs, gold_model, train_retain_loader, val_retain_loader, device = device, save_path = save_path)

print(f"Loading retrained model weights from {save_path}...")
gold_model.load_state_dict(torch.load(save_path))

# Evaluate retrained model
evaluate(gold_model, test_retain_loader, 'cuda')



# ----------------------------------------
# UNLEARNING METHODS
# ----------------------------------------

# Finetuning
print("Applying Fine-tuning method for unlearning...")
fntn_model = AllCNN(num_classes=1).to(device)
fntn_model.load_state_dict(torch.load("../saved_models/AllCNN_AgeDB_100epochs.pt"))
epochs = 5
save_path = "saved_models/AllCNN_AgeDB_5epochs_0to30_Finetune_Forget.pt"

history = fit_one_cycle(epochs, fntn_model, train_retain_loader, val_retain_loader, lr = 0.001, device = device, save_path = save_path)

evaluate(fntn_model, test_retain_loader, 'cuda')

evaluate(fntn_model, test_forget_loader, 'cuda')

## Gaussian-Amnesiac
print("Applying Gaussian-Amnesiac method for unlearning...")
mean = df_train['age'].mean()
sd = df_train['age'].std()

random_preds = np.random.normal(loc=mean, scale=sd, size=(len(df_train[df_train['age'] <= 30]),))

amnesiac_finetune_df = df_train.copy()
amnesiac_finetune_df.loc[amnesiac_finetune_df['age'] <=30, 'age'] = random_preds

amnesiac_finetune_train_data = AgeDB(data_dir='./data', df=amnesiac_finetune_df, img_size=32, split='train')
amnesiac_finetune_train_loader = DataLoader(amnesiac_finetune_train_data, batch_size=256, shuffle=True,
                              num_workers=64, pin_memory=True, drop_last=False)

amn_model = AllCNN(num_classes=1).to(device)
amn_model.load_state_dict(torch.load("../saved_models/AllCNN_AgeDB_100epochs.pt"))
epochs = 1
save_path = "saved_models/AllCNN_AgeDB_1epoch_0to30_Amnesiac_Finetune_Forget_tmp.pt"
history = fit_one_cycle(epochs, amn_model, amnesiac_finetune_train_loader, val_retain_loader, lr = 0.001, device = device, save_path = save_path)

evaluate(amn_model, test_retain_loader, 'cuda')

evaluate(amn_model, test_forget_loader, 'cuda')

## Blindspot Unlearning
warnings.filterwarnings('ignore')

df_train_forget['unlearn'] = 1
df_val_forget['unlearn'] = 1
df_test_forget['unlearn'] = 1
df_train_retain['unlearn'] = 0
df_val_retain['unlearn'] = 0
df_test_retain['unlearn'] = 0

udf_train = pd.concat([df_train_forget, df_train_retain])
utrain_data = UAgeDB(data_dir='./data', df=udf_train, img_size=32, split='train')
utrain_loader = DataLoader(utrain_data, batch_size=256, shuffle=True,
                              num_workers=64, pin_memory=True, drop_last=False)

# Making the blindspot model
device = 'cuda'
bdst_model = AllCNN(num_classes=1).to(device)
epochs = 2
save_path = "saved_models/AllCNN_AgeDB_2epochs_0to30_Proxy_tmp.pt"
history = fit_one_cycle(epochs, bdst_model, train_retain_loader, val_retain_loader, device = device, save_path = save_path)

#Obtaining the unlearned model
bdstu_model = AllCNN(num_classes=1).to(device)
bdstu_model.load_state_dict(torch.load("../saved_models/AllCNN_AgeDB_100epochs.pt"))
epochs = 1
save_path = "saved_models/AllCNN_AgeDB_0to30_1epochs_ATBeta_50_unlearn_tmp.pt"
history = fit_one_forget_cycle(epochs, bdstu_model, bdst_model,  utrain_loader, val_loader, lr = 0.001, device = device, save_path = save_path)

evaluate(bdstu_model, test_retain_loader, 'cuda')

evaluate(bdstu_model, test_forget_loader, 'cuda')



# ----------------------------------------
# COMPARING MODELS USING WASSERSTEIN DISTANCE
# ----------------------------------------

print("Comparing models with the gold model using Wasserstein distance...")

gold_predict = predict(gold_model, train_forget_loader, device)
gold_outputs = torch.squeeze(gold_predict).cpu().numpy()

# Original Model
full_predict = predict(model, train_forget_loader, device)
full_outputs = torch.squeeze(full_predict).cpu().numpy()

wasserstein_distance(full_outputs, gold_outputs)

# Finetune Model
fntn_predict = predict(fntn_model, train_forget_loader, device)
fntn_outputs = torch.squeeze(fntn_predict).cpu().numpy()

wasserstein_distance(fntn_outputs, gold_outputs)

# Gaussian Amnesiac Model
amn_predict = predict(amn_model, train_forget_loader, device)
amn_outputs = torch.squeeze(amn_predict).cpu().numpy()

wasserstein_distance(amn_outputs, gold_outputs)

# Blindspot Unlearned Model
bdstu_predict = predict(bdstu_model, train_forget_loader, device)
bdstu_outputs = torch.squeeze(bdstu_predict).cpu().numpy()

wasserstein_distance(bdstu_outputs, gold_outputs)



# ----------------------------------------
# COMPARING MODELS USING MEMBERSHIP INFERENCE PROBABILITY
# ----------------------------------------

print("Comparing models using Membership Inference Probability...")

sample_size = 2000
att_train_data = AgeDB(data_dir='./data', df=df_train.sample(sample_size), img_size=32, split='train')
att_val_data = AgeDB(data_dir='./data', df=df_val.sample(sample_size), img_size=32, split='val')
att_test_data = AgeDB(data_dir='./data', df=df_test.sample(sample_size), img_size=32, split='test')

att_train_loader = DataLoader(att_train_data, batch_size=256, shuffle=True,
                              num_workers=10, pin_memory=True, drop_last=False)
att_val_loader = DataLoader(att_val_data, batch_size=256, shuffle=False,
                            num_workers=10, pin_memory=True, drop_last=False)
att_test_loader = DataLoader(att_test_data, batch_size=256, shuffle=False,
                             num_workers=10, pin_memory=True, drop_last=False)

att_retain_data = AgeDB(data_dir='./data', df=df_train[df_train['age'] > 30].sample(sample_size), img_size=32, split='train')
att_forget_data = AgeDB(data_dir='./data', df=df_train[df_train['age'] <= 30].sample(sample_size), img_size=32, split='train')
att_forget_test_data = AgeDB(data_dir='./data', df=df_test[df_test['age'] <= 30].sample(min(sample_size, len(df_test[df_test['age'] <= 30]))), img_size=32, split='test')

att_forget_loader = DataLoader(att_forget_data, batch_size=256, shuffle=True,
                              num_workers=10, pin_memory=True, drop_last=False)
att_forget_test_loader = DataLoader(att_forget_test_data, batch_size=256, shuffle=True,
                              num_workers=10, pin_memory=True, drop_last=False)
att_retain_loader = DataLoader(att_retain_data, batch_size=256, shuffle=True,
                              num_workers=10, pin_memory=True, drop_last=False)

prediction_loaders = {"forget_data":att_forget_loader}

print("Original Model:")
get_membership_attack_prob(att_train_loader, att_test_loader, model, prediction_loaders)

print("Retrained Model:")
get_membership_attack_prob(att_train_loader, att_test_loader, gold_model, prediction_loaders)

print("Finetune Model:")
get_membership_attack_prob(att_train_loader, att_test_loader, fntn_model, prediction_loaders)

print("Gaussian Amnesiac Model:")
get_membership_attack_prob(att_train_loader, att_test_loader, amn_model, prediction_loaders)

print("Blindspot Unlearn Model:")
get_membership_attack_prob(att_train_loader, att_test_loader, bdstu_model, prediction_loaders)



# ----------------------------------------
# ADVERSARIAL INFLUENCE NEUTRALISATION
# ----------------------------------------

print("Comparing models with the gold model using Adversarial Influence Neutralization distance...")

print("Finetune Model:")
ain(model, fntn_model, gold_model, train_data, val_data_retain, val_data_forget, 
                  batch_size = 256, error_range = 0.05, lr = 0.01, device = 'cuda')

print("Gaussian Amnesiac Model:")
ain(model, amn_model, gold_model, train_data, val_data_retain, val_data_forget, 
                  batch_size = 256, error_range = 0.05, lr = 0.01, device = 'cuda')

print("Blindspot Unlearned Model:")
ain(model, bdstu_model, gold_model, train_data, val_data_retain, val_data_forget, 
                  batch_size = 256, error_range = 0.05, lr = 0.01, device = 'cuda')



# ----------------------------------------
# DISTRIBUTION COMPARISION
# ----------------------------------------

labels = df_train[df_train['age'] <= 30]['age'].values

pred_norm_dict = {'original model':abs(full_outputs-gold_outputs)/labels,
            'ours: blindspot':abs(bdstu_outputs-gold_outputs)/labels, 
            'finetune':abs(fntn_outputs-gold_outputs)/labels,
            'g-amnesiac':abs(amn_outputs-gold_outputs)/labels}

pred_norm_df = pd.DataFrame(pred_norm_dict)

plt.rcParams['font.size'] = '14'

sns.set_style("darkgrid")
sns.histplot(pred_norm_df, element="poly", stat='density')
plt.xlabel("Relative prediction difference from retrained model")
plt.xlim(0,2)