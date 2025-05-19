import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F
from torchvision.models import mobilenet_v3_large
from torchvision.models import resnet18

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_large()
        base_list = [*list(base.children())[:-1]]
        self.conv_norm1 = nn.Sequential(*base_list[0][0])
        for i in range(1, 16):
            exec(f"self.inverted_residual_{i} = base_list[0][{i}]")
        self.conv_norm2 = nn.Sequential(*base_list[0][16])
        self.pool1 = base_list[1]
        self.drop = nn.Dropout()
        self.final = nn.Linear(960,1)
    
    def forward(self,x):
        actvn1 = self.conv_norm1(x)
        
        for i in range(1, 16):
            exec(f"actvn{i+1} = self.inverted_residual_{i}(actvn{i})", locals(), globals())
        
        actvn17 = self.conv_norm2(actvn16)
        out = self.pool1(actvn17)
        
        out = self.drop(out.view(-1,self.final.in_features))
        return self.final(out), actvn1, actvn2, actvn3, actvn4, actvn5, actvn6, actvn7,\
                actvn8, actvn9, actvn10, actvn11, actvn12, actvn13, actvn14, actvn15,\
                actvn16, actvn17

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
    
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        return x.view(x.size(0), -1)

class ConvStandard(nn.Conv2d): 
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0, w_sig =\
                 np.sqrt(1.0)):
        super(ConvStandard, self).__init__(in_channels, out_channels,kernel_size)
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.kernel_size=kernel_size
        self.stride=stride
        self.padding=padding
        self.w_sig = w_sig
        self.reset_parameters()
      
    def reset_parameters(self):
        torch.nn.init.normal_(self.weight, mean=0, std=self.w_sig/(self.in_channels*np.prod(self.kernel_size)))
        if self.bias is not None:
            torch.nn.init.normal_(self.bias, mean=0, std=0)
            
    def forward(self, input):
        return F.conv2d(input,self.weight,self.bias,self.stride,self.padding)
            
class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, output_padding=0,
                 activation_fn=nn.ReLU, batch_norm=True, transpose=False):
        if padding is None:
            padding = (kernel_size - 1) // 2
        model = []
        if not transpose:
#             model += [ConvStandard(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
#                                 )]
            model += [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                bias=not batch_norm)]
        else:
            model += [nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                         output_padding=output_padding, bias=not batch_norm)]
        if batch_norm:
            model += [nn.BatchNorm2d(out_channels, affine=True)]
        model += [activation_fn()]
        super(Conv, self).__init__(*model)

class AllCNN(nn.Module):
    def __init__(self, filters_percentage=1., n_channels=3, num_classes=10, dropout=False, batch_norm=True):
        super(AllCNN, self).__init__()
        n_filter1 = int(96 * filters_percentage)
        n_filter2 = int(192 * filters_percentage)
        
        self.conv1 = Conv(n_channels, n_filter1, kernel_size=3, batch_norm=batch_norm)
        self.conv2 = Conv(n_filter1, n_filter1, kernel_size=3, batch_norm=batch_norm)
        self.conv3 = Conv(n_filter1, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)
        
        self.dropout1 = self.features = nn.Sequential(nn.Dropout(inplace=True) if dropout else Identity())
        
        self.conv4 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv5 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv6 = Conv(n_filter2, n_filter2, kernel_size=3, stride=2, padding=1, batch_norm=batch_norm)
        
        self.dropout2 = self.features = nn.Sequential(nn.Dropout(inplace=True) if dropout else Identity())
        
        self.conv7 = Conv(n_filter2, n_filter2, kernel_size=3, stride=1, batch_norm=batch_norm)
        self.conv8 = Conv(n_filter2, n_filter2, kernel_size=1, stride=1, batch_norm=batch_norm)
        self.pool = nn.AvgPool2d(8)
        self.flatten = Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(n_filter2, num_classes),
        )

    def forward(self, x):
        out = self.conv1(x)
        actv1 = out
        
        out = self.conv2(out)
        actv2 = out
        
        out = self.conv3(out)
        actv3 = out
        
        out = self.dropout1(out)
        
        out = self.conv4(out)
        actv4 = out
        
        out = self.conv5(out)
        actv5 = out
        
        out = self.conv6(out)
        actv6 = out
        
        out = self.dropout2(out)
        
        out = self.conv7(out)
        actv7 = out
        
        out = self.conv8(out)
        actv8 = out
        
        out = self.pool(out)
        
        out = self.flatten(out)
        
        out = self.classifier(out)
        
        return out, actv1, actv2, actv3, actv4, actv5, actv6, actv7, actv8 
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        base = resnet18(pretrained=False)
        in_features = base.fc.in_features
        base_list = [*list(base.children())[:-1]]
        self.layer1 = nn.Sequential(*base_list[0:3])
        self.pool1 = base_list[3]
        self.basic_block1 = base_list[4][0]
        self.basic_block2 = base_list[4][1]
        self.basic_block3 = base_list[5][0]
        self.basic_block4 = base_list[5][1]
        self.basic_block5 = base_list[6][0]
        self.basic_block6 = base_list[6][1]
        self.basic_block7 = base_list[7][0]
        self.basic_block8 = base_list[7][1]
        self.pool2 = base_list[8]
        self.drop = nn.Dropout()
        self.final = nn.Linear(512,1)
        
    
    def forward(self,x):
        out = self.layer1(x)
        actvn1 = out
        
        out = self.pool1(out)
        
        out = self.basic_block1(out)
        actvn2 = out
        
        out = self.basic_block2(out)
        actvn3 = out
        
        out = self.basic_block3(out)
        actvn4 = out
        
        out = self.basic_block4(out)
        actvn5 = out
        
        out = self.basic_block5(out)
        actvn6 = out
        
        out = self.basic_block6(out)
        actvn7 = out
        
        out = self.basic_block7(out)
        actvn8 = out
        
        out = self.basic_block8(out)
        actvn9 = out
        
        out = self.pool2(out)
        out = out.view(-1,self.final.in_features)
            
        out = self.final(out)
        
        return out, actvn1, actvn2, actvn3, actvn4, actvn5, actvn6, actvn7, actvn8, actvn9 

class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # --- START DEBUG PRINT ---
        # print(f"DEBUG TimeDistributed input: shape={x.shape}, dtype={x.dtype}, device={x.device}, numel={x.numel()}, " +
        #       f"module_type={type(self.module)}, batch_first_setting={self.batch_first}")
        # if isinstance(self.module, nn.Linear):
        #     print(f"DEBUG TimeDistributed nn.Linear module: in_features={self.module.in_features}, out_features={self.module.out_features}, bias_exists={self.module.bias is not None}")
        # elif isinstance(self.module, nn.BatchNorm1d):
        #     print(f"DEBUG TimeDistributed nn.BatchNorm1d module: num_features={self.module.num_features}")
        # elif isinstance(self.module, GLU):
        #      print(f"DEBUG TimeDistributed GLU module: fc1_in_features={self.module.fc1.in_features if hasattr(self.module, 'fc1') else 'N/A'}")
        # # --- END DEBUG PRINT ---

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # LINE 225 - ERROR HERE

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))
        else:
            y = y.view(-1, x.size(1), y.size(-1))

        return y
    
class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()
        
        self.fc1 = nn.Linear(input_size,input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)
    
class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size,hidden_state_size, output_size, dropout, hidden_context_size=None, batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        # print(f"DEBUG GRN Init: input_size={input_size}, hidden_state_size={hidden_state_size}, output_size={output_size}, " +
        #       f"hidden_context_size={hidden_context_size}, batch_first={batch_first}")
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size=hidden_state_size
        # self.dropout = dropout
        self.dropout_rate = dropout

        # Determine if this GRN will be a "pass-through" or modified operation due to zero sizes
        self.is_effectively_noop = (self.input_size == 0 and \
                                   (self.hidden_context_size is None or self.hidden_context_size == 0) and \
                                   self.output_size == 0)
        
        if self.input_size != self.output_size:
            if self.input_size == 0 and self.output_size > 0: # Input 0, Output > 0
                # Linear(0, K) only makes sense if bias=True, output is just bias
                # For safety, let's assume it should be an identity if input is 0 but output is expected to match
                # This case is tricky. If input_size=0, skip should ideally be a zero tensor of output_size
                # For now, we'll let Linear(0,K) be created if this happens.
                self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size, bias=True), batch_first=batch_first)
            elif self.input_size > 0 : # Normal case
                self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size), batch_first=batch_first)
            else: # input_size = 0 and output_size = 0
                self.skip_layer = None # Or Identity() if it needs to pass a (T,B,0) tensor
        else: # input_size == output_size
            self.skip_layer = None # Residual is x directly

        # FC1: input -> hidden
        # If input_size is 0, Linear(0, H) is problematic.
        # If input_size is 0, this path should effectively contribute zeros or be conditioned.
        if self.input_size > 0:
            self.fc1 = TimeDistributed(nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first)
        else: # input_size == 0
            # This fc1 will be Linear(0,H). It will be called with (T,B,0) if not careful.
            # For TimeDistributed to work, it needs to handle this.
            # The error occurs in TimeDistributed.view for this layer.
            # We MUST prevent Linear(0, H) if it will receive 0-element, 0-feature tensor.
            # This GRN should not have been created like this, or VSN should bypass it.
             self.fc1 = TimeDistributed(nn.Linear(self.input_size, self.hidden_state_size, bias=True), batch_first=batch_first) # Keep as is from your file

        self.elu1 = nn.ELU()
        
        if self.hidden_context_size is not None and self.hidden_context_size > 0:
            self.context_layer = TimeDistributed(nn.Linear(self.hidden_context_size, self.hidden_state_size), batch_first=batch_first)
        else:
            self.context_layer = None
                    
        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size,  self.output_size), batch_first=batch_first)
        # self.elu2 = nn.ELU()
        
        self.dropout_layer = nn.Dropout(self.dropout_rate) # Use stored rate
        
        if self.output_size > 0:
            self.bn = TimeDistributed(nn.BatchNorm1d(self.output_size), batch_first=batch_first)
            self.gate = TimeDistributed(GLU(self.output_size), batch_first=batch_first)
        else:
            self.bn = None
            self.gate = None

    def forward(self, x, context=None):

        if self.input_size!=self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x
        
        x = self.fc1(x)
        if context is not None:
            context = self.context_layer(context)
            x = x+context
        x = self.elu1(x)
        
        x = self.fc2(x)
        x = self.dropout_layer(x)
        x = self.gate(x)
        x = x+residual
        x = self.bn(x)
        
        return x

class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len,1,self.d_model)
            x = x + pe
            return x

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context_input_size=None):
        super(VariableSelectionNetwork, self).__init__()
        # print(f"DEBUG VSN Init: input_size(emb_dim)={input_size}, num_inputs(vars_to_select)={num_inputs}, " +
        #       f"hidden_size={hidden_size}, context_input_size={context_input_size}")
        
        self.hidden_size = hidden_size # This is H, the output dim of single_var_grns
        self.input_size = input_size   # This is emb_dim, features per variable
        self.num_variables_to_select = num_inputs # This is N_vars
        self.dropout_rate = dropout
        self.context_input_size = context_input_size
        
        # self.has_variables_to_select is true if there are variables AND each variable has features
        self.has_variables_to_select = self.num_variables_to_select > 0 and self.input_size > 0

        if self.has_variables_to_select:
            # Input to flattened_grn is all variable embeddings concatenated
            grn_input_features = self.num_variables_to_select * self.input_size 
            
            effective_context_size_for_flat_grn = self.context_input_size if self.context_input_size is not None and self.context_input_size > 0 else None

            self.flattened_grn = GatedResidualNetwork(
                grn_input_features, 
                self.hidden_size, # Hidden layer size for this GRN that calculates weights
                self.num_variables_to_select, # Output is N_vars weights
                self.dropout_rate, 
                effective_context_size_for_flat_grn 
            )
            self.softmax = nn.Softmax(dim=-1) 

            self.single_variable_grns = nn.ModuleList()
            for _ in range(self.num_variables_to_select):
                self.single_variable_grns.append(
                    GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout_rate) # Output is H
                )
        else: 
            # print(f"INFO: VSN (num_vars={self.num_variables_to_select}, emb_dim={self.input_size}) is NO-OP for variable selection part.")
            self.flattened_grn = None
            self.softmax = None
            self.single_variable_grns = nn.ModuleList() # Empty ModuleList

    def forward(self, embedding, context=None):
        # embedding: (timesteps, batch, features_for_variable_part = N_vars * emb_dim)
        # context: (timesteps, batch, features_for_context_part)
        # print(f"DEBUG VSN Forward: embedding_shape={embedding.shape if embedding is not None else 'None'}")
        # if context is not None: print(f"DEBUG VSN Forward: context_shape={context.shape}")

        timesteps = embedding.size(0)
        batch_size = embedding.size(1)

        if not self.has_variables_to_select:
            # If no variables to select (either num_inputs is 0 or input_size for vars is 0),
            # the VSN should still output a tensor of the expected hidden_size for downstream compatibility.
            # This output will be all zeros, representing no selected variable information.
            # print(f"DEBUG VSN Forward: No variables to select. Outputting zeros of shape ({timesteps},{batch_size},{self.hidden_size}).")
            return torch.zeros(timesteps, batch_size, self.hidden_size, 
                               device=embedding.device, dtype=embedding.dtype), None

        # --- Context processing for flattened_grn ---
        context_for_grn = None
        if self.context_input_size is not None and self.context_input_size > 0:
            if context is None:
                # print("Warning: VSN expected context for flattened_grn but got None in forward.")
                pass # flattened_grn's context input will be None
            elif context.size(-1) != self.context_input_size:
                raise ValueError(f"VSN Forward: Context feature mismatch for flattened_grn. Expected {self.context_input_size}, got {context.size(-1)}")
            else:
                context_for_grn = context
        
        # 1. Calculate variable selection weights
        # grn_output shape: (T, B, N_vars)
        grn_output = self.flattened_grn(embedding, context_for_grn) # 'embedding' here is the N_vars*emb_dim part
        # selection_probabilities shape: (T, B, N_vars)
        selection_probabilities = self.softmax(grn_output)

        # 2. Process each variable with its own GRN
        processed_vars_list = []
        for i in range(self.num_variables_to_select):
            # single_var_embedding_slice shape: (T, B, emb_dim)
            start_idx = i * self.input_size
            end_idx = (i + 1) * self.input_size
            single_var_embedding_slice = embedding[:, :, start_idx:end_idx]
            # processed_var shape: (T, B, H)
            processed_var = self.single_variable_grns[i](single_var_embedding_slice) # No context for single var GRNs by design
            processed_vars_list.append(processed_var) 

        # stacked_processed_vars shape: (T, B, H, N_vars)
        stacked_processed_vars = torch.stack(processed_vars_list, axis=-1) 
        
        # 3. Apply weights
        # Reshape selection_probabilities for broadcasting: from (T, B, N_vars) to (T, B, 1, N_vars)
        weights_for_broadcast = selection_probabilities.unsqueeze(2) 
        
        # Element-wise multiplication; weights_for_broadcast is broadcast across H dimension
        # weighted_vars shape: (T, B, H, N_vars)
        weighted_vars = stacked_processed_vars * weights_for_broadcast
        
        # Sum over the N_vars dimension to get aggregated features
        # selected_features shape: (T, B, H)
        selected_features = weighted_vars.sum(axis=-1) 

        # Return aggregated features and the (T, B, N_vars) selection probabilities
        return selected_features, selection_probabilities

class TFT(nn.Module):
    def __init__(self, config):
        super(TFT, self).__init__()
        self.device = config['device']
        # self.batch_size = config['batch_size'] # Store for reference if needed, but don't use for dynamic sizing
        self.static_variables = config['static_variables']
        self.encode_length = config['encode_length'] # Expected encoder length
        self.time_varying_categoical_variables =  config['time_varying_categoical_variables']
        self.time_varying_real_variables_encoder =  config['time_varying_real_variables_encoder']
        self.time_varying_real_variables_decoder =  config['time_varying_real_variables_decoder']
        self.num_input_series_to_mask = config['num_masked_series']
        self.hidden_size = config['lstm_hidden_dimension']
        self.lstm_layers = config['lstm_layers']
        self.dropout = config['dropout']
        self.embedding_dim = config['embedding_dim']
        self.attn_heads = config['attn_heads']
        self.num_quantiles = config['num_quantiles']
        # self.valid_quantiles = config['valid_quantiles'] # Not used in current code
        self.seq_length = config['seq_length'] # Total sequence length (encoder + decoder)

        # --- Embedding Layers ---
        self.static_embedding_layers = nn.ModuleList()
        for i in range(self.static_variables):
            emb = nn.Embedding(config['static_embedding_vocab_sizes'][i], config['embedding_dim'])
            self.static_embedding_layers.append(emb)

        self.time_varying_embedding_layers = nn.ModuleList()
        for i in range(self.time_varying_categoical_variables):
            emb = TimeDistributed(nn.Embedding(config['time_varying_embedding_vocab_sizes'][i], config['embedding_dim']), batch_first=True)
            self.time_varying_embedding_layers.append(emb)

        self.time_varying_linear_layers = nn.ModuleList()
        # Ensure this list covers all real variables (encoder + potentially unique decoder ones if indexed differently)
        # The current loop assumes all real vars are covered by time_varying_real_variables_encoder count for layer creation
        for i in range(self.time_varying_real_variables_encoder): # Or max of encoder/decoder real vars if they share layers by index
            emb = TimeDistributed(nn.Linear(1, config['embedding_dim']), batch_first=True)
            self.time_varying_linear_layers.append(emb)

        # --- Variable Selection Networks ---
        # Context for VSN is the concatenated static embeddings
        static_context_size = self.embedding_dim * self.static_variables if self.static_variables > 0 else None

        self.encoder_variable_selection = VariableSelectionNetwork(
            input_size=config['embedding_dim'],
            num_inputs=(config['time_varying_real_variables_encoder'] + config['time_varying_categoical_variables']),
            hidden_size=self.hidden_size, # VSN output feature dim
            dropout=self.dropout,
            context_input_size=static_context_size
        )

        self.decoder_variable_selection = VariableSelectionNetwork(
            input_size=config['embedding_dim'],
            num_inputs=(config['time_varying_real_variables_decoder'] + config['time_varying_categoical_variables']),
            hidden_size=self.hidden_size, # VSN output feature dim
            dropout=self.dropout,
            context_input_size=static_context_size
        )

        # --- LSTMs ---
        # Input to LSTM is the output of VSN (and Positional Encoding), which has self.hidden_size features
        self.lstm_encoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0 # Dropout only between LSTM layers
        )

        self.lstm_decoder = nn.LSTM(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=self.dropout if self.lstm_layers > 1 else 0 # Dropout only between LSTM layers
        )

        # --- Gating and Normalization Layers ---
        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size)) # batch_first=False by default for TD
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size)) # CORRECTED

        self.static_enrichment = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_state_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout,
            hidden_context_size=static_context_size # Static embeddings as context
        )

        self.position_encoding = PositionalEncoder(self.hidden_size, self.seq_length)

        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.attn_heads, dropout=self.dropout)
        self.post_attn_gate = TimeDistributed(GLU(self.hidden_size))
        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size)) # CORRECTED

        self.pos_wise_ff = GatedResidualNetwork(
            input_size=self.hidden_size,
            hidden_state_size=self.hidden_size,
            output_size=self.hidden_size,
            dropout=self.dropout
            # No context for this GRN in the standard TFT
        )

        self.pre_output_gate = TimeDistributed(GLU(self.hidden_size))
        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size)) # CORRECTED

        self.output_layer = TimeDistributed(nn.Linear(self.hidden_size, self.num_quantiles), batch_first=True)

        # Move all modules to device at the end of __init__
        self.to(self.device)


    def _init_hidden(self, current_batch_size): # Renamed to avoid conflict if subclassing
        # Returns h_0, c_0 for LSTM
        h_0 = torch.zeros(self.lstm_layers, current_batch_size, self.hidden_size, device=self.device)
        c_0 = torch.zeros(self.lstm_layers, current_batch_size, self.hidden_size, device=self.device)
        return h_0, c_0

    def apply_embedding(self, x_ts, static_embedding_ctx, num_real_vars, num_cat_vars, is_decoder_path):
        # x_ts: (batch_size, timesteps, num_features_for_this_path)
        # static_embedding_ctx: (batch_size, static_embedding_dim) or None
        # num_real_vars: number of real variables for this path (encoder/decoder)
        # num_cat_vars: number of categorical variables for this path (encoder/decoder)
        # is_decoder_path: boolean flag

        current_batch_size, num_timesteps, _ = x_ts.shape

        processed_real_embeddings = []
        if num_real_vars > 0:
            for i in range(num_real_vars):
                # Determine which column of x_ts and which layer to use
                # This logic needs to be robust based on how x_ts is structured
                # For simplicity, assume x_ts columns are ordered: real_var_0, real_var_1, ...
                # And time_varying_linear_layers correspond.
                # If masking, layer_idx might need adjustment.
                layer_idx_to_use = i
                col_idx_in_x_ts = i

                if is_decoder_path and self.num_input_series_to_mask > 0: # Example of masking logic
                    layer_idx_to_use = i + self.num_input_series_to_mask # If layers are shared and indexed globally
                    # col_idx_in_x_ts would still be 'i' if x_ts for decoder only contains decoder vars

                if layer_idx_to_use < len(self.time_varying_linear_layers) and col_idx_in_x_ts < x_ts.size(2):
                    emb = self.time_varying_linear_layers[layer_idx_to_use](
                        x_ts[:, :, col_idx_in_x_ts].unsqueeze(-1) # Shape (B, T, 1)
                    )
                    processed_real_embeddings.append(emb)
                # else: print warning or error about index out of bounds

        processed_cat_embeddings = []
        if num_cat_vars > 0:
            # Assume categorical variables in x_ts start after real variables
            for i in range(num_cat_vars):
                # This also assumes self.time_varying_embedding_layers are indexed 0 to num_cat_vars-1
                # And x_ts columns are real_0, ..., real_N-1, cat_0, ..., cat_M-1
                col_idx_in_x_ts = num_real_vars + i
                layer_idx_to_use = i # Assuming direct mapping for categorical layers

                if layer_idx_to_use < len(self.time_varying_embedding_layers) and col_idx_in_x_ts < x_ts.size(2):
                    emb = self.time_varying_embedding_layers[layer_idx_to_use](
                        x_ts[:, :, col_idx_in_x_ts].long() # Shape (B, T), TD Embedding handles unsqueeze
                    )
                    processed_cat_embeddings.append(emb)
                # else: print warning

        # Combine embeddings
        all_embeddings_for_vsn = []
        if processed_real_embeddings:
            all_embeddings_for_vsn.append(torch.cat(processed_real_embeddings, dim=2))
        if processed_cat_embeddings:
            all_embeddings_for_vsn.append(torch.cat(processed_cat_embeddings, dim=2))

        if not all_embeddings_for_vsn:
            # If no real or cat vars, input to VSN (variable part) is (B, T, 0)
            # VSN's `input_size` (emb_dim) is >0, but `num_inputs` might be 0.
            # VSN handles num_inputs=0 by outputting (T, B, H_vsn) zeros.
            # The `embedding` input to VSN's forward method will have 0 features.
            variable_embeddings_flat = torch.empty((current_batch_size, num_timesteps, 0), device=self.device, dtype=torch.float32)
        else:
            variable_embeddings_flat = torch.cat(all_embeddings_for_vsn, dim=2)
            # Shape: (batch, timesteps, num_selected_vars * embedding_dim)

        # Prepare static context for VSN
        static_context_for_vsn = None
        if static_embedding_ctx is not None and num_timesteps > 0:
            static_context_for_vsn = static_embedding_ctx.unsqueeze(1).repeat(1, num_timesteps, 1)
            # Shape: (batch, timesteps, static_embedding_dim)

        # Permute for VSN/LSTM: (timesteps, batch, features)
        variable_embeddings_flat = variable_embeddings_flat.permute(1, 0, 2).contiguous()
        if static_context_for_vsn is not None:
            static_context_for_vsn = static_context_for_vsn.permute(1, 0, 2).contiguous()

        return variable_embeddings_flat, static_context_for_vsn


    def forward(self, x_dict):
        # x_dict contains 'inputs': (batch, seq_len, num_features)
        #                 'identifier': (batch, seq_len, num_static_vars) - only first timestep used for statics

        input_ts = x_dict['inputs'].float().to(self.device) # (batch, total_seq_len, num_features)
        static_ids = x_dict['identifier'][:, 0, :].long().to(self.device) # (batch, num_static_vars)

        current_batch_size = input_ts.size(0)
        actual_total_seq_len = input_ts.size(1) # Could be different from self.seq_length if padding

        # --- 1. Static Embeddings ---
        static_embedding_vectors = []
        if self.static_variables > 0:
            for i in range(self.static_variables):
                emb = self.static_embedding_layers[i](static_ids[:, i])
                static_embedding_vectors.append(emb)
        
        static_embeddings_cat = None
        if static_embedding_vectors:
            static_embeddings_cat = torch.cat(static_embedding_vectors, dim=1) # (batch, total_static_emb_dim)


        # --- 2. Prepare inputs for Encoder & Decoder paths ---
        # Encoder inputs
        encoder_input_ts = input_ts[:, :self.encode_length, :] # (B, enc_len, N_feat_enc)
        # Decoder inputs (known future inputs)
        # Ensure actual_total_seq_len > self.encode_length for decoder_input_ts to be valid
        if actual_total_seq_len > self.encode_length:
            decoder_input_ts = input_ts[:, self.encode_length:, :] # (B, dec_len, N_feat_dec)
        else: # Handle cases where only encoder data is present or seq_len is too short
            # Create an empty tensor for decoder_input_ts if no decoder steps
            # This assumes decoder_len could be 0
            decoder_seq_len = actual_total_seq_len - self.encode_length
            if decoder_seq_len <=0 :
                 # Create a placeholder for decoder_input_ts if no actual decoder timesteps
                decoder_input_ts = torch.empty(current_batch_size, 0, input_ts.size(2), device=self.device, dtype=input_ts.dtype)
            else: # Should not happen if actual_total_seq_len > self.encode_length
                decoder_input_ts = input_ts[:, self.encode_length:, :]


        # --- 3. Embed known inputs & Apply VSN ---
        # Encoder
        encoder_vars_emb, encoder_static_ctx_repeated = self.apply_embedding(
            encoder_input_ts, static_embeddings_cat,
            self.time_varying_real_variables_encoder,
            self.time_varying_categoical_variables, # Assuming cat vars are same count for enc/dec
            is_decoder_path=False
        )
        # encoder_vars_emb: (enc_len, B, N_enc_selected_vars * emb_dim)
        # encoder_static_ctx_repeated: (enc_len, B, total_static_emb_dim)
        
        processed_encoder_input, _ = self.encoder_variable_selection(
            encoder_vars_emb,
            encoder_static_ctx_repeated
        ) # Output: (enc_len, B, hidden_size)

        # Decoder
        # Check if decoder_input_ts has any timesteps
        if decoder_input_ts.size(1) > 0 :
            decoder_vars_emb, decoder_static_ctx_repeated = self.apply_embedding(
                decoder_input_ts, static_embeddings_cat,
                self.time_varying_real_variables_decoder,
                self.time_varying_categoical_variables, # Assuming cat vars are same count for enc/dec
                is_decoder_path=True
            )
            processed_decoder_input, _ = self.decoder_variable_selection(
                decoder_vars_emb,
                decoder_static_ctx_repeated
            ) # Output: (dec_len, B, hidden_size)
        else: # No decoder timesteps
            processed_decoder_input = torch.empty(0, current_batch_size, self.hidden_size, device=self.device, dtype=processed_encoder_input.dtype)


        # --- 4. Positional Encoding ---
        # Create positional encoding for the maximum possible sequence length used by this batch
        # The PE module's max_seq_len should be >= actual_total_seq_len
        # If PE is generated for self.seq_length, ensure actual_total_seq_len <= self.seq_length
        effective_seq_len_for_pe = processed_encoder_input.size(0) + processed_decoder_input.size(0)
        if effective_seq_len_for_pe > 0:
            pe_base = torch.zeros(effective_seq_len_for_pe, current_batch_size, self.hidden_size, device=self.device)
            pos_enc = self.position_encoding(pe_base) # (eff_seq_len, B, hidden_size)

            # Add PE
            # Actual encoder length from tensor shape
            actual_encoder_len = processed_encoder_input.size(0)
            if actual_encoder_len > 0:
                processed_encoder_input = processed_encoder_input + pos_enc[:actual_encoder_len, :, :]
            
            if processed_decoder_input.size(0) > 0: # If there are decoder steps
                 processed_decoder_input = processed_decoder_input + pos_enc[actual_encoder_len:, :, :]
        

        # --- 5. LSTM Encoder-Decoder ---
        # Encoder
        initial_h_encoder, initial_c_encoder = self._init_hidden(current_batch_size)
        if processed_encoder_input.size(0) > 0: # Only run LSTM if there are encoder steps
            encoder_lstm_output, (encoder_h_n, encoder_c_n) = self.lstm_encoder(
                processed_encoder_input, (initial_h_encoder, initial_c_encoder)
            )
        else: # No encoder steps, prepare zero states for decoder if needed
            encoder_lstm_output = torch.empty(0, current_batch_size, self.hidden_size, device=self.device, dtype=processed_encoder_input.dtype)
            encoder_h_n, encoder_c_n = initial_h_encoder, initial_c_encoder # Use initial zeros

        # Decoder
        if processed_decoder_input.size(0) > 0: # Only run LSTM if there are decoder steps
            decoder_lstm_output, _ = self.lstm_decoder(
                processed_decoder_input, (encoder_h_n, encoder_c_n) # Use encoder's final state
            )
        else:
            decoder_lstm_output = torch.empty(0, current_batch_size, self.hidden_size, device=self.device, dtype=processed_decoder_input.dtype)

        # Concatenate LSTM outputs
        # lstm_output: (total_eff_seq_len, B, hidden_size)
        lstm_output_combined = torch.cat([encoder_lstm_output, decoder_lstm_output], dim=0)

        # --- 6. Post-LSTM Gating & Normalization (Skip connection) ---
        # lstm_input_for_skip is the concatenated VSN outputs + PE
        lstm_input_for_skip = torch.cat([processed_encoder_input, processed_decoder_input], dim=0)
        
        gated_lstm_output = self.post_lstm_gate(lstm_output_combined + lstm_input_for_skip)
        normed_gated_lstm_output = self.post_lstm_norm(gated_lstm_output)


        # --- 7. Static Enrichment ---
        # static_embeddings_cat: (B, total_static_emb_dim)
        # normed_gated_lstm_output: (total_eff_seq_len, B, hidden_size)
        if static_embeddings_cat is not None and normed_gated_lstm_output.size(0) > 0 :
            static_ctx_for_enrichment = static_embeddings_cat.unsqueeze(0).repeat(normed_gated_lstm_output.size(0), 1, 1)
            # static_ctx_for_enrichment: (total_eff_seq_len, B, total_static_emb_dim)
            enriched_output = self.static_enrichment(normed_gated_lstm_output, static_ctx_for_enrichment)
        else: # No static features or no LSTM output to enrich
            enriched_output = normed_gated_lstm_output


        # --- 8. Self-Attention ---
        # Attention is applied only to the decoder part, using encoder part as memory
        # actual_encoder_len was calculated from processed_encoder_input.size(0)
        
        # Query from decoder part of enriched_output
        # Key/Value from encoder part of enriched_output
        if enriched_output.size(0) > actual_encoder_len and actual_encoder_len > 0 : # Need both encoder and decoder parts
            query_tensor = enriched_output[actual_encoder_len:, :, :] # (dec_len, B, hidden_size)
            key_tensor   = enriched_output[:actual_encoder_len, :, :] # (enc_len, B, hidden_size)
            value_tensor = enriched_output[:actual_encoder_len, :, :] # (enc_len, B, hidden_size)

            attn_output_raw, attn_weights = self.multihead_attn(query_tensor, key_tensor, value_tensor)
            # attn_output_raw: (dec_len, B, hidden_size)

            # --- 9. Post-Attention Gating & Normalization (Skip connection) ---
            # Skip connection adds the query tensor (decoder part before attention)
            gated_attn_output = self.post_attn_gate(attn_output_raw + query_tensor)
            normed_attn_output = self.post_attn_norm(gated_attn_output)
            
            # --- 10. Position-wise Feed-Forward ---
            ff_output = self.pos_wise_ff(normed_attn_output) # (dec_len, B, hidden_size)

            # --- 11. Final Gating & Normalization (Skip connection) ---
            # Skip connection adds the decoder part of the LSTM output (after its own skip+gate)
            # Make sure decoder_lstm_output has content if we are here
            if decoder_lstm_output.size(0) > 0:
                 # We need the gated & normed version of decoder LSTM output for consistency.
                 # This would be normed_gated_lstm_output[actual_encoder_len:, :, :]
                decoder_lstm_for_skip = normed_gated_lstm_output[actual_encoder_len:, :, :]
                final_decoder_output = self.pre_output_gate(ff_output + decoder_lstm_for_skip)
                final_decoder_output_normed = self.pre_output_norm(final_decoder_output)
            else: # Should not happen if ff_output has content based on decoder steps
                final_decoder_output_normed = ff_output # Or handle error
        
        elif enriched_output.size(0) > 0 and actual_encoder_len == enriched_output.size(0) :
            # Only encoder was processed, no decoder part for attention.
            # This means the output should likely be empty or handled as an edge case.
            # For now, assume if we reach here, we expect decoder predictions.
            # If this path is valid, 'output' needs to be defined.
            # This situation implies no decoder timesteps, so prediction is tricky.
            # Let's assume the task requires decoder timesteps for prediction.
            # If not, the logic for 'final_decoder_output_normed' needs to be adapted.
            # For now, create empty tensor if no decoder path through attention.
             final_decoder_output_normed = torch.empty(0, current_batch_size, self.hidden_size, device=self.device, dtype=enriched_output.dtype)
        
        else: # No content from LSTM/enrichment, or no encoder part for attention keys/values
            final_decoder_output_normed = torch.empty(0, current_batch_size, self.hidden_size, device=self.device, dtype=enriched_output.dtype)


        # --- 12. Output Layer ---
        # Reshape for TimeDistributed Linear with batch_first=True
        # final_decoder_output_normed: (dec_len, B, hidden_size)
        if final_decoder_output_normed.size(0) > 0:
            output_layer_input = final_decoder_output_normed.permute(1, 0, 2) # (B, dec_len, hidden_size)
            predictions = self.output_layer(output_layer_input) # (B, dec_len, num_quantiles)
        else: # No decoder timesteps processed to this point
            predictions = torch.empty(current_batch_size, 0, self.num_quantiles, device=self.device, dtype=final_decoder_output_normed.dtype)

        # For consistency, let's always return all named outputs, even if some are empty
        # based on control flow. The training loop/loss calculation needs to handle this.
        # The original return signature included encoder_output, decoder_output etc.
        # These are intermediate and might not always be fully populated if seq lengths are short.

        # Placeholder for sparse weights if you need them for debugging/analysis
        encoder_sparse_weights = None # Not explicitly returned by VSN in this simplified flow
        decoder_sparse_weights = None

        return predictions #, encoder_lstm_output, decoder_lstm_output, attn_output_raw, attn_weights, encoder_sparse_weights, decoder_sparse_weights
    
class LSTMnetwork(nn.Module):
    def __init__(self, text_embedding_dimension):
        super().__init__()
        self.hidden_size = 64
        self.input_size = text_embedding_dimension
        self.num_layers = 1
        self.bidirectional = False
        self.num_directions = 1
        self.dropout1 = nn.Dropout(p=0.3)

        if self.bidirectional:
            self.num_directions = 2
 
        self.lstm = nn.LSTM( self.input_size, self.hidden_size, self.num_layers, 
                             bidirectional=self.bidirectional, batch_first=True)
        
        self.linear1 = nn.Linear(self.hidden_size*self.num_directions*2, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 16)
        self.linear4 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, sent1, sent2):
        
        lstm_out1, _ = self.lstm( sent1)

        x1 = self.dropout1( lstm_out1)
        
        actv1 = x1
        
        lstm_out2, _ = self.lstm( sent2)
        
        x2 = self.dropout1( lstm_out2)
        
        actv2 = x2
        
        output = self.linear1(torch.cat([x1[:, -1, :], x2[:, -1, :]], axis = 1))
        actv3 = output
        output = self.relu(output)
        
        
        output = self.linear2(output)
        actv4 = output
        output = self.relu(output)
        
        
        output = self.linear3(output)
        output = self.relu(output)
        output = self.linear4(output)
        
        return torch.squeeze(output), actv1, actv2, actv3, actv4