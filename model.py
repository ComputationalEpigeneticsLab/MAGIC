
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.ao.nn.quantized import LeakyReLU

#torch.autograd.set_detect_anomaly(True)
from torch_geometric.nn import global_mean_pool, global_max_pool, GlobalAttention
from transMIL_wsi import TransMIL_wsi
from transMIL_hovernet import TransMIL_hovernet


#
class Attn_Modality_Gated(nn.Module):
    # Adapted from https://github.com/mahmoodlab/PORPOISE
    def __init__(self, gate_h1, gate_h2,gate_h3, dim1_og, dim2_og, dim3_og, use_bilinear=[True, True,True], scale=[1, 1, 1], p_dropout_fc=0.25):
        super(Attn_Modality_Gated, self).__init__()

        self.gate_h1 = gate_h1  # [boolean]
        self.gate_h2 = gate_h2  # [boolean]
        self.gate_h3 = gate_h3  # [boolean]
        self.use_bilinear = use_bilinear  # [boolean]

        # can perform attention on latent vectors of lower dimension
        dim1, dim2, dim3 = dim1_og // scale[0], dim2_og // scale[1], dim3_og // scale[2]

        # attention gate of each modality
        if self.gate_h1:
            self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
            self.linear_z1 = nn.Bilinear(dim1_og, dim2_og+dim3_og, dim1) if self.use_bilinear[0] else nn.Sequential(nn.Linear(dim1_og + dim2_og + dim3_og, dim1))
            self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h1, self.linear_o1 = nn.Identity(), nn.Identity()

        if self.gate_h2:
            self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
            self.linear_z2 = nn.Bilinear(dim2_og,  dim1_og+dim3_og, dim2) if self.use_bilinear[1] else nn.Sequential(nn.Linear(dim1_og + dim2_og + dim3_og, dim2))
            self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h2, self.linear_o2 = nn.Identity(), nn.Identity()


        if self.gate_h3:
            self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
            self.linear_z3 = nn.Bilinear(dim3_og, dim1_og+dim2_og, dim3) if self.use_bilinear[2] else nn.Sequential(nn.Linear(dim1_og + dim2_og + dim3_og, dim3))
            self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h3, self.linear_o3 = nn.Identity(), nn.Identity()


    def forward(self, x1, x2, x3):
        if self.gate_h1:
            h1 = self.linear_h1(x1)
            z1 = self.linear_z1(x1, torch.cat([x2,x3], dim=-1)) if self.use_bilinear[0] else self.linear_z1(torch.cat((x1, x2,x3), dim=-1))  # creates a vector combining both modalities
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(x1)
            o1 = self.linear_o1(h1)

        if self.gate_h2:
            h2 = self.linear_h2(x2)
            z2 = self.linear_z2(x2, torch.cat([x1,x3], dim=-1)) if self.use_bilinear[1] else self.linear_z2(torch.cat((x1, x2,x3), dim=-1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(x2)
            o2 = self.linear_o2(h2)

        if self.gate_h3:
            h3 = self.linear_h3(x3)
            z3 = self.linear_z3(x3, torch.cat([x1,x2], dim=-1)) if self.use_bilinear[2] else self.linear_z3(torch.cat((x1, x2,x3), dim=-1))
            o3 = self.linear_o3(nn.Sigmoid()(z3) * h3)
        else:
            h3 = self.linear_h3(x3)
            o3 = self.linear_o3(h3)

        modality_weights = {
            'WSI': torch.sigmoid(z1).mean().item(),
            'HoVerNet': torch.sigmoid(z2).mean().item(),
            'Protein': torch.sigmoid(z3).mean().item()
        }

        return o1, o2, o3,modality_weights


#
class Attn_Modality_Gated_two_model(nn.Module):
    # Adapted from https://github.com/mahmoodlab/PORPOISE
    def __init__(self, gate_h1, gate_h2, dim1_og, dim2_og, use_bilinear=[True, True], scale=[1, 1], p_dropout_fc=0.25):
        super(Attn_Modality_Gated_two_model, self).__init__()

        self.gate_h1 = gate_h1  # [boolean]
        self.gate_h2 = gate_h2  # [boolean]
        self.use_bilinear = use_bilinear  # [boolean]

        # can perform attention on latent vectors of lower dimension
        dim1, dim2 = dim1_og // scale[0], dim2_og // scale[1]

        # attention gate of each modality
        if self.gate_h1:
            self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
            self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if self.use_bilinear[0] else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim1))
            self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h1, self.linear_o1 = nn.Identity(), nn.Identity()

        if self.gate_h2:
            self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
            self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if self.use_bilinear[1] else nn.Sequential(nn.Linear(dim1_og + dim2_og, dim2))
            self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=p_dropout_fc))
        else:
            self.linear_h2, self.linear_o2 = nn.Identity(), nn.Identity()


    def forward(self, x1, x2):
        if self.gate_h1:
            h1 = self.linear_h1(x1)
            z1 = self.linear_z1(x1, x2) if self.use_bilinear[0] else self.linear_z1(torch.cat((x1, x2), dim=-1))  # creates a vector combining both modalities
            o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)
        else:
            h1 = self.linear_h1(x1)
            o1 = self.linear_o1(h1)

        if self.gate_h2:
            h2 = self.linear_h2(x2)
            z2 = self.linear_z2(x1, x2) if self.use_bilinear[1] else self.linear_z2(torch.cat((x1, x2), dim=-1))
            o2 = self.linear_o2(nn.Sigmoid()(z2) * h2)
        else:
            h2 = self.linear_h2(x2)
            o2 = self.linear_o2(h2)


        return o1, o2



class MLP_Block(nn.Module):
    def __init__(self, dim_in, hidden_dims: list, p_dropout_fc=0.25):
        super(MLP_Block, self).__init__()
        mlp_layers = []
        in_features = dim_in
        for h_dim in hidden_dims:
            mlp_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.InstanceNorm1d(h_dim),
                    nn.LeakyReLU(),
                    nn.Dropout(p=p_dropout_fc, inplace=False)
                )
            )
            in_features = h_dim

        self.block = nn.Sequential(*mlp_layers)

    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(0)
        return x


class SNN_Block(nn.Module):
    def __init__(self, dim_in, hidden_dims: list, p_dropout_fc=0.25):
        super(SNN_Block, self).__init__()
        snn_layers = []
        in_features = dim_in
        for h_dim in hidden_dims:
            snn_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.InstanceNorm1d(h_dim),
                    nn.ELU(),
                    nn.AlphaDropout(p=p_dropout_fc, inplace=False)
                )
            )
            in_features = h_dim

        self.block = nn.Sequential(*snn_layers)


    def forward(self, x):
        x = self.block(x)
        x = x.squeeze(0)
        return x


class VAE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int, dropout_prob: float = 0.5):
        """
        Variational Autoencoder (VAE) model.

        Args:
            input_dim (int): Dimension of the input data (protein expression data).
            hidden_dims (list): List of dimensions for the hidden layers of the encoder and decoder.
            latent_dim (int): Dimension of the latent space.
            dropout_prob (float): Dropout probability for the input layer.
        """
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder
        encoder_layers = [nn.Sequential(nn.Dropout(dropout_prob))]
        in_features = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.InstanceNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            in_features = h_dim
        self.encoder = nn.Sequential(*encoder_layers)

        # Latent space mean and log variance
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # Decoder
        decoder_layers = []
        in_features = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(
                nn.Sequential(
                    nn.Linear(in_features, h_dim),
                    nn.InstanceNorm1d(h_dim),
                    nn.ReLU()
                )
            )
            in_features = h_dim
        decoder_layers.append(nn.Sequential(nn.Linear(hidden_dims[0], input_dim)))
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x):

        h = self.encoder(x)
        z_mu = self.fc_mu(h)
        z_logvar = self.fc_logvar(h)
        return z_mu, z_logvar

    def reparameterize(self, z_mu, z_logvar):
        """
        Reparameterization trick to sample from the latent space.

        Args:
            z_mu (Tensor): Mean of the latent space.
            z_logvar (Tensor): Log variance of the latent space.

        Returns:
            z (Tensor): Sampled latent vector.
        """
        std = torch.exp(0.5 * z_logvar)
        eps = torch.randn_like(std)
        z = z_mu + eps * std
        return z

    def decode(self, z):

        x_recon = self.decoder(z)
        return x_recon

    def forward(self, x):
        z_mu, z_logvar = self.encode(x)
        z = self.reparameterize(z_mu, z_logvar)
        x_recon = self.decode(z)
        return x_recon.squeeze(0), z_mu.squeeze(0), z_logvar.squeeze(0)

#
class FC_block(nn.Module):
    def __init__(self, dim_in, dim_out, act_layer=nn.ReLU, dropout=True, p_dropout_fc=0.25):
        super(FC_block, self).__init__()

        self.fc = nn.Linear(dim_in, dim_out)
        self.act = act_layer()
        self.drop = nn.Dropout(p_dropout_fc) if dropout else nn.Identity()

    def forward(self, x):
        x = self.fc(x)
        x = self.act(x)
        x = self.drop(x)
        return x

#
class FusionModel(nn.Module):
    def __init__(
            self,
            patch_feature_size=384,
            dropout_atten_mil=True,
            hovernet_feature_size = 171,
            prot_input_size=180,
            p_dropout_fc=0.20,
            vae_hidden_dims=[128],
            vae_latent_dims=64,
            vae_dropout = 0.2,
            gate_hist=True,
            gate_hovernet=True,
            gate_prot=True,
            use_bilinear=[True, True, True],
            scale_hist=1,
            scale_hovernet=1,
            scale_prot=1):
        super(FusionModel,self).__init__()
        #
        self.patch_embedding_uni = TransMIL_wsi(patch_feature_dim =patch_feature_size)

        #
        self.patch_embedding_hovernet = TransMIL_hovernet(patch_feature_dim =hovernet_feature_size)

        #
        self.prot_embedding = VAE(input_dim=prot_input_size, hidden_dims=vae_hidden_dims, latent_dim=vae_latent_dims,
                                  dropout_prob=vae_dropout)
        self.post_compression_layer_prot = nn.Sequential(
            *[FC_block(vae_latent_dims, vae_latent_dims, p_dropout_fc=p_dropout_fc)])
        #

        self.attn_modality_gated = Attn_Modality_Gated(gate_h1=gate_hist, gate_h2=gate_hovernet, gate_h3=gate_prot, dim1_og=512,
                                                       dim2_og=171, dim3_og=vae_latent_dims, use_bilinear=use_bilinear,
                                                       scale=[scale_hist, scale_hovernet,scale_prot])
        #
        dim_post_compression_wsi = 512 // scale_hist if gate_hist else 512
        self.post_compression_layer_uni = nn.Sequential(*[FC_block(dim_post_compression_wsi, 256, p_dropout_fc=p_dropout_fc),
                                                          FC_block(256, 64, p_dropout_fc=p_dropout_fc)])
        #
        dim_post_compression_hovernet = 171 // scale_hovernet if gate_hovernet else 171
        self.post_compression_layer_hovernet = nn.Sequential(
            *[FC_block(dim_post_compression_hovernet, 64, p_dropout_fc=p_dropout_fc)])
        #

        self.concat_size = 64*3
        self.post_compression_layer = nn.Sequential(*[FC_block(self.concat_size, 64, p_dropout_fc=p_dropout_fc)])
        self.classifier = nn.Sequential(nn.Linear(64, 1),nn.Sigmoid())

        self.apply(self._init_weights)


    #
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


    def forward(self, wsi_feats, morph_feats, protein_feats,type='train'):
        if type == 'test':
            patches_feature_uni, attn_matrix_uni, all_feature_uni = self.patch_embedding_uni(wsi_feats,return_attention=True)

            patches_feature_hovernet, attn_matrix_hovernet, all_feature_hovernet  = self.patch_embedding_hovernet(morph_feats, return_attention=True)

            x_recon, prot_feature_vae, log_var = self.prot_embedding(protein_feats)
            prot_feature = self.post_compression_layer_prot(prot_feature_vae)
            #
            # Attention gates on each modality.
            patches_feature_uni, patches_feature_hovernet, prot_feature,modality_weights = self.attn_modality_gated(patches_feature_uni, patches_feature_hovernet, prot_feature)
            #
            patches_feature_uni = self.post_compression_layer_uni(patches_feature_uni)
            patches_feature_hovernet = self.post_compression_layer_hovernet(patches_feature_hovernet)
            #

            fusion_feature = torch.cat([patches_feature_uni, patches_feature_hovernet,prot_feature], dim=-1)
            fusion_feature = self.post_compression_layer(fusion_feature)
            #
            fused_prob = self.classifier(fusion_feature)

            return fused_prob, fusion_feature, x_recon, prot_feature_vae, log_var, all_feature_uni, all_feature_hovernet, attn_matrix_uni, attn_matrix_hovernet,modality_weights
        else:
            patches_feature_uni = self.patch_embedding_uni(wsi_feats)
            #
            patches_feature_hovernet = self.patch_embedding_hovernet(morph_feats)
            #
            x_recon, prot_feature_vae, log_var = self.prot_embedding(protein_feats)
            prot_feature = self.post_compression_layer_prot(prot_feature_vae)
            #
            # Attention gates on each modality.
            patches_feature_uni, patches_feature_hovernet, prot_feature,_ = self.attn_modality_gated(patches_feature_uni,
                                                                                                   patches_feature_hovernet,
                                                                                                   prot_feature)
            #
            patches_feature_uni = self.post_compression_layer_uni(patches_feature_uni)
            patches_feature_hovernet = self.post_compression_layer_hovernet(patches_feature_hovernet)
            #

            fusion_feature = torch.cat([patches_feature_uni, patches_feature_hovernet, prot_feature], dim=-1)
            fusion_feature = self.post_compression_layer(fusion_feature)
            #
            fused_prob = self.classifier(fusion_feature)

            return fused_prob, fusion_feature, x_recon, prot_feature_vae, log_var

#
class Only_WSI_UNI(nn.Module):
    def __init__(
            self,
            patch_feature_size=384,
            dropout_atten_mil=True,
            p_dropout_fc=0.25
    ):
        super(Only_WSI_UNI, self).__init__()
        self.patch_embedding =TransMIL_wsi(patch_feature_dim =patch_feature_size)

        self.post_compression_layer_he = nn.Sequential(*[FC_block(512, 256, p_dropout_fc=p_dropout_fc),
                                                         FC_block(256, 64, p_dropout_fc=p_dropout_fc)])
        feature_size_comp_post = 64

        self.classifier = nn.Sequential(nn.Linear(feature_size_comp_post, 1),nn.Sigmoid())
        self.apply(self._init_weights)

    #
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # InstanceNorm1d
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


    def forward(self, hist):
        wsi_feature = self.patch_embedding(hist)
        #
        wsi_feature = self.post_compression_layer_he(wsi_feature)
        # predict
        output = self.classifier(wsi_feature)

        return output, wsi_feature


class Only_WSI_hovernet(nn.Module):
    def __init__(
            self,
            hovernet_feature_size=171,
            dropout_atten_mil=True,
            p_dropout_fc=0.25
    ):
        super(Only_WSI_hovernet, self).__init__()
        self.patch_embedding =TransMIL_hovernet(patch_feature_dim =hovernet_feature_size)
        self.post_compression_layer_he = nn.Sequential(*[FC_block(171, 64, p_dropout_fc=p_dropout_fc)])

        feature_size_comp_post = 64

        self.classifier = nn.Sequential(nn.Linear(feature_size_comp_post, 1),nn.Sigmoid())
        self.apply(self._init_weights)

    #
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # InstanceNorm1d
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


    def forward(self, hist):
        wsi_feature = self.patch_embedding(hist)
        #
        wsi_feature = self.post_compression_layer_he(wsi_feature)
        # predict
        output = self.classifier(wsi_feature)
        return output, wsi_feature


class Only_Prot(nn.Module):
    def __init__(
            self,
            prot_embedd_method='vae',
            prot_input_size=180,
            prot_hidden_dims=[64,64],
            p_dropout_fc=0.2,
            vae_hidden_dims=[128],
            vae_latent_dims=64,
            vae_dropout=0.2
    ):
        super(Only_Prot, self).__init__()
        self.prot_embedd_method = prot_embedd_method

        if self.prot_embedd_method == 'mlp':
            self.prot_embedding = MLP_Block(dim_in=prot_input_size, hidden_dims=prot_hidden_dims, p_dropout_fc=p_dropout_fc)
            self.prot_feature_size = prot_hidden_dims[-1]

        elif self.prot_embedd_method == 'snn':
            self.prot_embedding = SNN_Block(dim_in=prot_input_size, hidden_dims=prot_hidden_dims, p_dropout_fc=p_dropout_fc)
            self.prot_feature_size = prot_hidden_dims[-1]

        elif self.prot_embedd_method == 'vae':
            self.prot_embedding = VAE(input_dim=prot_input_size, hidden_dims=vae_hidden_dims, latent_dim=vae_latent_dims, dropout_prob=vae_dropout)
            self.post_compression_layer_prot = nn.Sequential(
                *[FC_block(vae_latent_dims, vae_latent_dims, p_dropout_fc=p_dropout_fc)])
            self.prot_feature_size = vae_latent_dims

        feature_size_comp_post = self.prot_feature_size

        self.classifier = nn.Sequential(nn.Linear(feature_size_comp_post, 1),nn.Sigmoid())
        self.apply(self._init_weights)

    #
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


    def forward(self, prot):
        if self.prot_embedd_method == 'vae':
            x_recon, prot_feature_vae, log_var = self.prot_embedding(prot)
            prot_feature_final = self.post_compression_layer_prot(prot_feature_vae)
            output = self.classifier(prot_feature_final)
            return output, x_recon, prot_feature_vae, log_var
        else:
            prot_feature = self.prot_embedding(prot)
            # predict
            output = self.classifier(prot_feature)
            return output


class FusionModel_UNI_hovernet(nn.Module):
    def __init__(
            self,
            patch_feature_size=384,
            dropout_atten_mil=True,
            hovernet_feature_size=171,
            p_dropout_fc=0.20,
            gate_hist=True,
            gate_hovernet=True,
            use_bilinear=[True, True],
            scale_hist=1,
            scale_hovernet=1):
        super(FusionModel_UNI_hovernet,self).__init__()
        #
        self.patch_embedding_uni = TransMIL_wsi(patch_feature_dim =patch_feature_size)

        #
        self.patch_embedding_hovernet = TransMIL_hovernet(patch_feature_dim =hovernet_feature_size)

        #

        self.attn_modality_gated = Attn_Modality_Gated_two_model(gate_h1=gate_hist, gate_h2=gate_hovernet,
                                                                 dim1_og=512,
                                                                 dim2_og=171,
                                                                 use_bilinear=use_bilinear,
                                                                 scale=[scale_hist, scale_hovernet])
        #
        dim_post_compression_wsi = 512 // scale_hist if gate_hist else 512
        self.post_compression_layer_uni = nn.Sequential(*[FC_block(dim_post_compression_wsi, 256, p_dropout_fc=p_dropout_fc),
                                                          FC_block(256, 64, p_dropout_fc=p_dropout_fc)])
        #
        dim_post_compression_hovernet = 171 // scale_hovernet if gate_hovernet else 171
        self.post_compression_layer_hovernet = nn.Sequential(
            *[FC_block(dim_post_compression_hovernet, 64, p_dropout_fc=p_dropout_fc)])
        #

        self.concat_size = 64*2
        self.post_compression_layer = nn.Sequential(*[FC_block(self.concat_size, 64, p_dropout_fc=p_dropout_fc)])
        self.classifier = nn.Sequential(nn.Linear(64, 1),nn.Sigmoid())

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)


    def forward(self, wsi_feats, morph_feats):
        patches_feature_uni = self.patch_embedding_uni(wsi_feats)
        #
        patches_feature_hovernet = self.patch_embedding_hovernet(morph_feats)
        #

        # Attention gates on each modality.
        patches_feature_uni, patches_feature_hovernet = self.attn_modality_gated(patches_feature_uni, patches_feature_hovernet)
        #
        patches_feature_uni = self.post_compression_layer_uni(patches_feature_uni)
        patches_feature_hovernet = self.post_compression_layer_hovernet(patches_feature_hovernet)
        #

        fusion_feature = torch.cat([patches_feature_uni, patches_feature_hovernet], dim=-1)
        fusion_feature = self.post_compression_layer(fusion_feature)
        #
        fused_prob = self.classifier(fusion_feature)

        return fused_prob, fusion_feature, patches_feature_uni, patches_feature_hovernet


class FusionModel_UNI_prot(nn.Module):
    def __init__(
            self,
            patch_feature_size=384,
            dropout_atten_mil=True,
            prot_input_size=180,
            p_dropout_fc=0.20,
            vae_hidden_dims=[128],
            vae_latent_dims=64,
            vae_dropout=0.2,
            gate_hist=True,
            gate_prot=True,
            use_bilinear=[True, True],
            scale_hist=1,
            scale_prot=1):
        super(FusionModel_UNI_prot, self).__init__()
        #
        self.patch_embedding_uni = TransMIL_wsi(patch_feature_dim =patch_feature_size)

        #
        self.prot_embedding = VAE(input_dim=prot_input_size, hidden_dims=vae_hidden_dims, latent_dim=vae_latent_dims,
                                  dropout_prob=vae_dropout)
        self.post_compression_layer_prot = nn.Sequential(
            *[FC_block(vae_latent_dims, vae_latent_dims, p_dropout_fc=p_dropout_fc)])
        #

        self.attn_modality_gated = Attn_Modality_Gated_two_model(gate_h1=gate_hist, gate_h2=gate_prot,
                                                       dim1_og=512,dim2_og=vae_latent_dims,
                                                       use_bilinear=use_bilinear,scale=[scale_hist, scale_prot])
        #
        dim_post_compression_wsi = 512 // scale_hist if gate_hist else 512
        self.post_compression_layer_uni = nn.Sequential(
            *[FC_block(dim_post_compression_wsi, 256, p_dropout_fc=p_dropout_fc),
              FC_block(256, 64, p_dropout_fc=p_dropout_fc)])

        #
        self.concat_size = 64 * 2
        self.post_compression_layer = nn.Sequential(*[FC_block(self.concat_size, 64, p_dropout_fc=p_dropout_fc)])
        self.classifier = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # InstanceNorm1d
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)



    def forward(self, wsi_feats, protein_feats):
        patches_feature_uni = self.patch_embedding_uni(wsi_feats)

        #
        x_recon, prot_feature_vae, log_var = self.prot_embedding(protein_feats)
        prot_feature = self.post_compression_layer_prot(prot_feature_vae)
        #
        # Attention gates on each modality.
        patches_feature_uni, prot_feature = self.attn_modality_gated(patches_feature_uni,prot_feature)
        #
        patches_feature_uni = self.post_compression_layer_uni(patches_feature_uni)
        #
        fusion_feature = torch.cat([patches_feature_uni, prot_feature], dim=-1)
        fusion_feature = self.post_compression_layer(fusion_feature)
        #
        fused_prob = self.classifier(fusion_feature)

        return fused_prob, fusion_feature, x_recon, prot_feature_vae, log_var, patches_feature_uni


class FusionModel_hovernet_prot(nn.Module):
    def __init__(
            self,
            dropout_atten_mil=True,
            hovernet_feature_size=171,
            prot_input_size=180,
            p_dropout_fc=0.20,
            vae_hidden_dims=[128],
            vae_latent_dims=64,
            vae_dropout=0.2,
            gate_hovernet=True,
            gate_prot=True,
            use_bilinear=[True, True],
            scale_hovernet=1,
            scale_prot=1):
        super(FusionModel_hovernet_prot, self).__init__()

        #
        self.patch_embedding_hovernet =  TransMIL_hovernet(patch_feature_dim =hovernet_feature_size)

        #
        self.prot_embedding = VAE(input_dim=prot_input_size, hidden_dims=vae_hidden_dims, latent_dim=vae_latent_dims,
                                  dropout_prob=vae_dropout)
        self.post_compression_layer_prot = nn.Sequential(
            *[FC_block(vae_latent_dims, vae_latent_dims, p_dropout_fc=p_dropout_fc)])
        #

        self.attn_modality_gated = Attn_Modality_Gated_two_model(gate_h1=gate_hovernet, gate_h2=gate_prot,
                                                       dim1_og=171, dim2_og=vae_latent_dims,
                                                       use_bilinear=use_bilinear,
                                                       scale=[scale_hovernet, scale_prot])
        #
        #
        dim_post_compression_hovernet = 171 // scale_hovernet if gate_hovernet else 171
        self.post_compression_layer_hovernet = nn.Sequential(
            *[FC_block(dim_post_compression_hovernet, 64, p_dropout_fc=p_dropout_fc)])
        #

        self.concat_size = 64 * 2
        self.post_compression_layer = nn.Sequential(*[FC_block(self.concat_size, 64, p_dropout_fc=p_dropout_fc)])
        self.classifier = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())

        self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        # InstanceNorm1d
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self,morph_feats, protein_feats):
        #
        patches_feature_hovernet = self.patch_embedding_hovernet(morph_feats)
        #
        x_recon, prot_feature_vae, log_var = self.prot_embedding(protein_feats)
        prot_feature = self.post_compression_layer_prot(prot_feature_vae)
        #
        # Attention gates on each modality.
        patches_feature_hovernet, prot_feature = self.attn_modality_gated(patches_feature_hovernet,prot_feature)
        #
        patches_feature_hovernet = self.post_compression_layer_hovernet(patches_feature_hovernet)
        #
        fusion_feature = torch.cat([patches_feature_hovernet, prot_feature], dim=-1)
        fusion_feature = self.post_compression_layer(fusion_feature)
        #
        fused_prob = self.classifier(fusion_feature)

        return fused_prob, fusion_feature, x_recon, prot_feature_vae, log_var, patches_feature_hovernet

