import torch
import os
import wget
import zipfile
import re

import numpy as np
import pandas as pd
from torch.nn import functional as F
from utils import clean_text
from models import LSTMnetwork

def training_step(model, batch, device):
    sent1, sent2, labels = batch 
    sent1, sent2, labels = sent1.to(device), sent2.to(device), labels.to(device)
    out, *_  = model(sent1, sent2)                  # Generate predictions
    loss= F.mse_loss(out, labels) # Calculate loss
    return loss

def validation_step(model, batch, device):
    sent1, sent2, labels = batch 
    sent1, sent2, labels = sent1.to(device), sent2.to(device), labels.to(device)
    out, *_  = model(sent1, sent2)                    # Generate predictions
    loss= F.mse_loss(out, labels)   # Calculate loss
    return {'Loss': loss.detach()}

def validation_epoch_end(model, outputs):
    batch_losses = [x['Loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    return {'Loss': epoch_loss.item()}

def epoch_end(model, epoch, result):
    print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}".format(
        epoch, result['lrs'][-1], result['train_loss'], result['Loss']))



@torch.no_grad()
def evaluate(model, val_df, device, batch_size = 256):
    model.eval()
    outputs = []
    
    num_steps = len(val_df)//batch_size
    
    for i in range(num_steps):
        sent1 = torch.tensor(np.stack(val_df.iloc[i*batch_size:(i+1)*batch_size]['sentence1'])).float()
        sent2 = torch.tensor(np.stack(val_df.iloc[i*batch_size:(i+1)*batch_size]['sentence2'])).float()
        labels = torch.tensor(val_df.iloc[i*batch_size:(i+1)*batch_size]['score'].values)
        batch = (sent1, sent2, labels)

        outputs.append(validation_step(model, batch, device))
        
    if len(val_df)%batch_size != 0:
        sent1 = torch.tensor(np.stack(val_df.iloc[num_steps*batch_size:]['sentence1'])).float()
        sent2 = torch.tensor(np.stack(val_df.iloc[num_steps*batch_size:]['sentence2'])).float()
        labels = torch.tensor(val_df.iloc[num_steps*batch_size:]['score'].values)
        batch = (sent1, sent2, labels)

        outputs.append(validation_step(model, batch, device))
        
    return validation_epoch_end(model, outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs,  model, train_df, val_df, device, save_path, batch_size = 256):
    best_loss = np.inf
    torch.cuda.empty_cache()
    history = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    num_steps = len(train_df)//batch_size
    
    #for 
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        lrs = []
        for i in range(num_steps):
            sent1 = torch.tensor(np.stack(train_df.iloc[i*batch_size:(i+1)*batch_size]['sentence1'])).float()
            sent2 = torch.tensor(np.stack(train_df.iloc[i*batch_size:(i+1)*batch_size]['sentence2'])).float()
            labels = torch.tensor(train_df.iloc[i*batch_size:(i+1)*batch_size]['score'].values).float()
            batch = (sent1, sent2, labels)
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
        if len(train_df)%batch_size != 0:
            sent1 = torch.tensor(np.stack(train_df.iloc[num_steps*batch_size:]['sentence1'])).float()
            sent2 = torch.tensor(np.stack(train_df.iloc[num_steps*batch_size:]['sentence2'])).float()
            labels = torch.tensor(train_df.iloc[num_steps*batch_size:]['score'].values).float()
            batch = (sent1, sent2, labels)
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
        
        # Validation phase
        result = evaluate(model, val_df, device, batch_size)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        sched.step(result['Loss'])
        if best_loss > result['Loss']:
            best_loss = result['Loss']
            torch.save(model.state_dict(), save_path)
    
    return history

def fit_one_finetune_cycle(epochs,  model, train_df, val_df, lr, device, save_path, batch_size = 256):
    best_loss = np.inf
    torch.cuda.empty_cache()
    history = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    num_steps = len(train_df)//batch_size
    
    #for 
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        lrs = []
        for i in range(num_steps):
            sent1 = torch.tensor(np.stack(train_df.iloc[i*batch_size:(i+1)*batch_size]['sentence1'])).float()
            sent2 = torch.tensor(np.stack(train_df.iloc[i*batch_size:(i+1)*batch_size]['sentence2'])).float()
            labels = torch.tensor(train_df.iloc[i*batch_size:(i+1)*batch_size]['score'].values).float()
            batch = (sent1, sent2, labels)
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
        if len(train_df)%batch_size != 0:
            sent1 = torch.tensor(np.stack(train_df.iloc[num_steps*batch_size:]['sentence1'])).float()
            sent2 = torch.tensor(np.stack(train_df.iloc[num_steps*batch_size:]['sentence2'])).float()
            labels = torch.tensor(train_df.iloc[num_steps*batch_size:]['score'].values).float()
            batch = (sent1, sent2, labels)
            loss = training_step(model, batch, device)
            train_losses.append(loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
        
        # Validation phase
        result = evaluate(model, val_df, device, batch_size)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        sched.step(result['Loss'])
        if best_loss > result['Loss']:
            best_loss = result['Loss']
            torch.save(model.state_dict(), save_path)
    
    return history

def attention(x):
        """
        Taken from https://github.com/szagoruyko/attention-transfer
        :param x = activations
        """
        return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def attention_diff(x, y):
    """
    Taken from https://github.com/szagoruyko/attention-transfer
    :param x = activations
    :param y = activations
    """
    return (attention(x) - attention(y)).pow(2).mean()



def forget_loss(model_output, model_activations, proxy_output, proxy_activations, mask):

    loss = F.mse_loss(model_output[mask], proxy_output[mask])
    if AT_beta > 0:
        at_loss = 0
        for i in range(len(proxy_activations)):
            at_loss = at_loss + AT_beta * attention_diff(model_activations[i][mask], proxy_activations[i][mask])
    else:
        at_loss = 0

    total_loss = loss + at_loss

    return total_loss



def fit_one_forget_cycle(epochs,  model, proxy_model, train_df, val_df, lr, device, save_path, batch_size = 256):
    best_loss = np.inf
    torch.cuda.empty_cache()
    history = []
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    num_steps = len(train_df)//batch_size
    for epoch in range(epochs): 
        model.train()
        train_losses = []
        lrs = []
        #for batch in train_loader:
        for i in range(num_steps):
            sent1 = torch.tensor(np.stack(train_df.iloc[i*batch_size:(i+1)*batch_size]['sentence1'])).float()
            sent2 = torch.tensor(np.stack(train_df.iloc[i*batch_size:(i+1)*batch_size]['sentence2'])).float()
            labels = torch.tensor(train_df.iloc[i*batch_size:(i+1)*batch_size]['score'].values).float()
            ulabels = torch.tensor(train_df.iloc[i*batch_size:(i+1)*batch_size]['forget'].values)
            
            sent1, sent2, labels, ulabels = sent1.to(device), sent2.to(device), labels.to(device), ulabels.to(device)
            
            model_out, *model_activations = model(sent1, sent2)
            with torch.no_grad():
                proxy_out, *proxy_activations = proxy_model(sent1, sent2)
                
            
            label_loss = 0
            if ulabels.sum() < len(ulabels):
                mask = (ulabels == 0)
                r_model_out = model_out[mask]
                r_labels = labels[mask]
                label_loss = F.mse_loss(r_model_out, r_labels)
            
            proxy_loss = 0
            if ulabels.sum() > 0:
                mask = (ulabels == 1)
                proxy_loss = forget_loss(model_out, model_activations, proxy_out, proxy_activations, mask)
            
            coeff = ulabels.sum()/len(ulabels)
            loss = coeff*proxy_loss + (1-coeff)*label_loss
            
            ######
            train_losses.append(loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
        if len(train_df)%batch_size != 0:
            sent1 = torch.tensor(np.stack(train_df.iloc[num_steps*batch_size:]['sentence1'])).float()
            sent2 = torch.tensor(np.stack(train_df.iloc[num_steps*batch_size:]['sentence2'])).float()
            labels = torch.tensor(train_df.iloc[num_steps*batch_size:]['score'].values).float()
            ulabels = torch.tensor(train_df.iloc[num_steps*batch_size:]['forget'].values)
            
            sent1, sent2, labels, ulabels = sent1.to(device), sent2.to(device), labels.to(device), ulabels.to(device)
            
            model_out, *model_activations = model(sent1, sent2)
            with torch.no_grad():
                proxy_out, *proxy_activations = proxy_model(sent1, sent2)
                
            
            label_loss = 0
            if ulabels.sum() < len(ulabels):
                mask = (ulabels == 0)
                r_model_out = model_out[mask]
                r_labels = labels[mask]
                label_loss = F.mse_loss(r_model_out, r_labels)
            
            proxy_loss = 0
            if ulabels.sum() > 0:
                mask = (ulabels == 1)
                proxy_loss = forget_loss(model_out, model_activations, proxy_out, proxy_activations, mask)
            
            coeff = ulabels.sum()/len(ulabels)
            loss = coeff*proxy_loss + (1-coeff)*label_loss
            
            ######
            train_losses.append(loss)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            
        
        # Validation phase
        result = evaluate(model, val_df, device)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        epoch_end(model, epoch, result)
        history.append(result)
        #sched.step(result['Loss'])
        #if best_loss > result['Loss']:
        #    best_loss = result['Loss']
        #    torch.save(model.state_dict(), save_path)
        torch.save(model.state_dict(), save_path)
    
    return history

text_embedding_dimension = 300

def text_embed(words):
    
    unknown_indices = []
    mean = np.zeros(text_embedding_dimension)
    
    for i in range(len(words)):
        if words[i] in embeddings_index_300 and embeddings_index_300[ words[i] ].shape == (300, ):
            words[i] = embeddings_index_300[ words[i] ]
            mean += words[i]
        else:
            unknown_indices.append(i)
            
    mean /= max(len(words)-len(unknown_indices), 1)
    
    # unknown words in the text are represented using the mean of the known words
    for i in unknown_indices:
        words[i] = mean
    return words

def pad(x, max_len = 10):
    if len(x) >= max_len:
        return x[:10]
    zeros = [np.zeros(text_embedding_dimension)]*(max_len - len(x))

# Get GLOVE Word embedding
print("Downloading and extracting GloVe word embeddings...")
data_file = "./glove.840B.300d.zip"
wget.download("http://nlp.stanford.edu/data/glove.840B.300d.zip", out=data_file)
with zipfile.ZipFile(data_file) as zip_ref:
    zip_ref.extractall('./glove')
os.remove(data_file)
print("\nCompleted!")

path_to_glove_file = "./glove/glove.840B.300d.txt"

embeddings_index_300 = {}
with open(path_to_glove_file) as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_index_300[word] = coefs

print("Found %s word vectors." % len(embeddings_index_300))

# Load data
train_df = pd.read_csv("./stsb_data/train_new.tsv", sep='\t', on_bad_lines='skip')
val_df = pd.read_csv("./stsb_data/dev_new.tsv", sep='\t', on_bad_lines='skip')
test_df = pd.read_csv("./stsb_data/test_new.tsv", sep='\t', on_bad_lines='skip')

train_df.dropna(subset=['score'], inplace=True)
val_df.dropna(subset=['score'], inplace=True)
test_df.dropna(subset=['score'], inplace=True)

train_df['sentence1'] = train_df['sentence1'].apply(lambda x: clean_text(x))
train_df['sentence2'] = train_df['sentence2'].apply(lambda x: clean_text(x))

val_df['sentence1'] = val_df['sentence1'].apply(lambda x: clean_text(x))
val_df['sentence2'] = val_df['sentence2'].apply(lambda x: clean_text(x))

test_df['sentence1'] = test_df['sentence1'].apply(lambda x: clean_text(x))
test_df['sentence2'] = test_df['sentence2'].apply(lambda x: clean_text(x))

train_df['sentence1'] = train_df['sentence1'].apply(lambda words: text_embed(words))
train_df['sentence2'] = train_df['sentence2'].apply(lambda words: text_embed(words))

val_df['sentence1'] = val_df['sentence1'].apply(lambda words: text_embed(words))
val_df['sentence2'] = val_df['sentence2'].apply(lambda words: text_embed(words))

test_df['sentence1'] = test_df['sentence1'].apply(lambda words: text_embed(words))
test_df['sentence2'] = test_df['sentence2'].apply(lambda words: text_embed(words))

train_df['sentence1'] = train_df['sentence1'].apply(lambda words: pad(words))
train_df['sentence2'] = train_df['sentence2'].apply(lambda words: pad(words))

val_df['sentence1'] = val_df['sentence1'].apply(lambda words: pad(words))
val_df['sentence2'] = val_df['sentence2'].apply(lambda words: pad(words))

test_df['sentence1'] = test_df['sentence1'].apply(lambda words: pad(words))
test_df['sentence2'] = test_df['sentence2'].apply(lambda words: pad(words))

train_df = train_df.sample(frac = 1, random_state = 0)

# Train the model
device = 'cuda'
model = LSTMnetwork(text_embedding_dimension = text_embedding_dimension).to(device)
epochs = 100
save_path = "saved_models/LSTM_STSB_100epochs.pt"
history = fit_one_cycle(epochs, model, train_df, val_df, device = device, save_path = save_path)
model.load_state_dict(torch.load(save_path))

# Creating the forget and retain sets
train_df_retain = train_df[train_df['score'] >= 2]
val_df_retain = val_df[val_df['score'] >= 2]
test_df_retain = test_df[test_df['score'] >= 2]

train_df_forget = train_df[train_df['score'] < 2]
val_df_forget = val_df[val_df['score'] < 2]
test_df_forget = test_df[test_df['score'] < 2]

# Retraining the model from scratch on Retain Data
device = 'cuda'
gold_model = LSTMnetwork(text_embedding_dimension = text_embedding_dimension).to(device)

epochs = 100
save_path = "saved_models/LSTM_STSB_100epochs_0to2_retrained.pt"
history = fit_one_cycle(epochs, gold_model, train_df_retain, val_df_retain, device = device, save_path = save_path)
gold_model.load_state_dict(torch.load(save_path))

# Evaluate the retrained model on various cohorts
evaluate(model, test_df_retain, 'cuda')
evaluate(model, test_df_forget, 'cuda')
evaluate(gold_model, test_df_retain, 'cuda')
evaluate(gold_model, test_df_forget, 'cuda')

# Finetuning
%%time
student_model = LSTMnetwork(text_embedding_dimension = text_embedding_dimension).to(device)
student_model.load_state_dict(torch.load("saved_models/LSTM_STSB_100epochs.pt"))
epochs = 5
save_path = "saved_models/LSTM_STSB_5epochs_4to5_Finetune_Forget.pt"
history = fit_one_finetune_cycle(epochs, student_model, train_df_retain, val_df_retain, 0.001, device = device, save_path = save_path)
student_model.load_state_dict(torch.load(save_path))

# Amnesiac Finetuning
mean = train_df['score'].mean()
sd = train_df['score'].std()

random_preds = np.random.normal(loc=mean, scale=sd, size=(len(train_df[train_df['score'] < 2]),))

amnesiac_finetune_df = train_df.copy()
amnesiac_finetune_df.loc[amnesiac_finetune_df['score'] < 2, 'score'] = random_preds

%%time
student_model = LSTMnetwork(text_embedding_dimension = text_embedding_dimension).to(device)
student_model.load_state_dict(torch.load("saved_models/LSTM_STSB_100epochs.pt"))
epochs = 5
save_path = "saved_models/LSTM_STSB_2epochs_Amnesiac_Finetune_Forget.pt"
history = fit_one_finetune_cycle(epochs, student_model, amnesiac_finetune_df, val_df_retain, 0.001, device = device, save_path = save_path)
student_model.load_state_dict(torch.load(save_path))

evaluate(student_model, test_df_retain, 'cuda')
evaluate(student_model, test_df_forget, 'cuda')

# Blindspot unlearning
u_train_df = train_df.copy()
u_train_df['forget'] = 0
u_train_df.loc[u_train_df['score'] < 2, 'forget'] = 1

# Training the Blindspot model
%%time
device = 'cuda'
proxy_model = LSTMnetwork(text_embedding_dimension = text_embedding_dimension).to(device)
epochs = 10
save_path = "saved_models/LSTM_STSB_blindspot.pt"
history = fit_one_cycle(epochs, proxy_model, train_df_retain, val_df_retain, device = device, save_path = save_path)
proxy_model.load_state_dict(torch.load(save_path))

evaluate(proxy_model, test_df_retain, 'cuda')
evaluate(proxy_model, test_df_forget, 'cuda')

%%time
AT_beta = 50
student_model = LSTMnetwork(text_embedding_dimension = text_embedding_dimension).to(device)
student_model.load_state_dict(torch.load("saved_models/LSTM_STSB_100epochs.pt"))
epochs = 5
save_path = "saved_models/LSTM_STSB_unlearn.pt"
history = fit_one_forget_cycle(epochs, student_model, proxy_model,  u_train_df, val_df, lr = 0.001, device = device, save_path = save_path)
student_model.load_state_dict(torch.load(save_path))

evaluate(student_model, test_df_retain, 'cuda')
evaluate(student_model, test_df_forget, 'cuda')
