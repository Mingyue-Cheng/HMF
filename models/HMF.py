import torch
import torch.nn as nn
from layers.Embed import patchDataEmbedding
from layers.Transformer_EncDec import series_decomp, Decoder , DecoderonlyLayer, transConv
from layers.SelfAttention_Family import AttentionLayer, FullAttention

class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.stride = configs.conv_stride
        self.actual_len = configs.seq_len + configs.pred_len
        self.seq_conv_len = ((configs.seq_len + 2 * configs.conv_padding - configs.conv_kernel) // configs.conv_stride) + 1
        self.pred_conv_len = ((configs.pred_len + 2 * configs.conv_padding - configs.conv_kernel) // configs.conv_stride) + 1
        self.device = torch.device(f'cuda:{configs.gpu}' if configs.use_gpu else 'cpu')
        self.patch_stride = self.stride // configs.patch_num
        self.patch_kernel = self.patch_stride
        self.conv_kernel = self.stride
        self.conv_stride = self.stride
        self.patch_padding = 0
        self.patch_num = configs.patch_num
        self.patch_stride = self.stride // self.patch_num
        self.patch_kernel = self.patch_stride
        self.patch_conv_len =((self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
        self.conv_len = ((self.actual_len + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
        self.enc_decompsition = series_decomp(configs.moving_avg)
        self.dec_decompsition = series_decomp(configs.moving_avg)
        self.train_autoregression = configs.train_autoregression


        self.seasonal_embedding = patchDataEmbedding(configs.enc_in, configs.d_model, self.patch_stride, self.patch_kernel, self.patch_padding, 
                                           configs.dropout, configs.grad, self.conv_len, configs.freq, configs.embed)
        self.seasonal_decoder = Decoder(
                [
                    DecoderonlyLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.enc_in, bias=True)
            )
        self.seasonal_trans_patch = transConv(self.patch_conv_len, configs.conv_stride, configs.dropout)

        self.trend_embedding = patchDataEmbedding(configs.enc_in, configs.d_model, self.patch_stride, self.patch_kernel, self.patch_padding, 
                                           configs.dropout, configs.grad, self.conv_len, configs.freq, configs.embed)
        self.trend_decoder = Decoder(
                [
                    DecoderonlyLayer(
                        AttentionLayer(
                            FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
                projection=nn.Linear(configs.d_model, configs.enc_in, bias=True)
            )
        self.trend_trans_patch = transConv(self.patch_conv_len, configs.conv_stride, configs.dropout)
        
    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        
        seasonal_init_enc, trend_init_enc = self.enc_decompsition(x_enc)
        seasonal_init_dec, trend_init_dec = self.dec_decompsition(x_dec)

        #seasonal
        seasonal_means = seasonal_init_enc.mean(1, keepdim=True).detach()
        seasonal_init_enc = seasonal_init_enc - seasonal_means
        seasonal_stdev = torch.sqrt(
            torch.var(seasonal_init_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        seasonal_init_enc /= seasonal_stdev
        seasonal_init_dec = seasonal_init_dec - seasonal_means
        seasonal_init_dec /= seasonal_stdev
        seasonal_input = torch.cat((seasonal_init_enc, seasonal_init_dec[:, -self.pred_len:, :]), dim=1)
        seasonal_output = torch.zeros([x_enc.size(0), self.actual_len, x_enc.size(2)], device=self.device)
        seasonal_output[:, :self.stride, :] = seasonal_init_enc[:, :self.stride, :]
        for i in range(self.seq_conv_len - 1):
            seasonal_zeros = torch.zeros([x_enc.size(0), self.actual_len - self.stride * (i + 1), x_enc.size(2)], device=self.device)
            seasonal_embedding_input = torch.cat((seasonal_input[:, :self.stride * (i + 1), :], seasonal_zeros), dim = 1)
            seasonal_embedding = self.seasonal_embedding(seasonal_embedding_input)
            start = ((i *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
            end = (((i + 1) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
            seasonal_embedding = seasonal_embedding[: , start : end , :]
            seasonal_current_output = self.seasonal_decoder(seasonal_embedding, seasonal_embedding)
            seasonal_current_output = self.seasonal_trans_patch(seasonal_current_output)
            seasonal_output[:, self.stride * (i + 1) : self.stride * (i + 2) , :] = seasonal_current_output 
        
        for i in range(self.pred_conv_len):
            seasonal_zeros = torch.zeros([x_enc.size(0), self.actual_len - self.stride * i - self.seq_len, x_enc.size(2)], device=self.device)
            if self.training and (not self.train_autoregression):
                seasonal_embedding_input = torch.cat((seasonal_init_enc, seasonal_input[:, self.seq_len: self.seq_len + i * self.stride, :], seasonal_zeros), dim = 1)
                seasonal_embedding = self.seasonal_embedding(seasonal_embedding_input)
                start = (((self.seq_conv_len + i - 1) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
                end = (((self.seq_conv_len + i) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
                seasonal_embedding = seasonal_embedding[: , start :end, :]
                seasonal_current_output = self.seasonal_decoder(seasonal_embedding, seasonal_embedding)
                seasonal_current_output = self.seasonal_trans_patch(seasonal_current_output)
                seasonal_output[:, self.seq_len + self.stride * i : self.seq_len + self.stride * (i + 1) , :] = seasonal_current_output 
            else:
                seasonal_embedding_input = torch.cat((seasonal_init_enc, seasonal_output[:, self.seq_len: self.seq_len + i * self.stride, :], seasonal_zeros), dim = 1)
                seasonal_embedding = self.seasonal_embedding(seasonal_embedding_input)
                start = (((self.seq_conv_len + i - 1) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
                end = (((self.seq_conv_len + i) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
                seasonal_embedding = seasonal_embedding[: , start :end, :]
                seasonal_current_output = self.seasonal_decoder(seasonal_embedding, seasonal_embedding)
                seasonal_current_output = self.seasonal_trans_patch(seasonal_current_output)
                seasonal_output[:, self.seq_len + self.stride * i : self.seq_len + self.stride * (i + 1) , :] = seasonal_current_output 
        seasonal_output = seasonal_output * \
                  (seasonal_stdev[:, 0, :].unsqueeze(1).repeat(1, self.actual_len, 1))
        seasonal_output = seasonal_output + \
                  (seasonal_means[:, 0, :].unsqueeze(1).repeat(1,  self.actual_len, 1))
        

        #trend
        trend_means = trend_init_enc.mean(1, keepdim=True).detach()
        trend_init_enc = trend_init_enc - trend_means
        trend_stdev = torch.sqrt(
            torch.var(trend_init_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        trend_init_enc /= trend_stdev
        trend_init_dec = trend_init_dec - trend_means
        trend_init_dec /= trend_stdev
        trend_input = torch.cat((trend_init_enc, trend_init_dec[:, -self.pred_len:, :]), dim=1)
        trend_output = torch.zeros([x_enc.size(0), self.actual_len, x_enc.size(2)], device=self.device)
        trend_output[:, :self.stride, :] = trend_init_enc[:, :self.stride, :]
        for i in range(self.seq_conv_len - 1):
            trend_zeros = torch.zeros([x_enc.size(0), self.actual_len - self.stride * (i + 1), x_enc.size(2)], device=self.device)
            trend_embedding_input = torch.cat((trend_input[:, :self.stride * (i + 1), :], trend_zeros), dim = 1)
            trend_embedding = self.trend_embedding(trend_embedding_input)
            start = ((i *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
            end = (((i + 1) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
            trend_embedding = trend_embedding[: , start : end , :]
            trend_current_output = self.trend_decoder(trend_embedding, trend_embedding)
            trend_current_output = self.trend_trans_patch(trend_current_output)
            trend_output[:, self.stride * (i + 1) : self.stride * (i + 2) , :] = trend_current_output 
        
        for i in range(self.pred_conv_len):
            trend_zeros = torch.zeros([x_enc.size(0), self.actual_len - self.stride * i - self.seq_len, x_enc.size(2)], device=self.device)
            if self.training and (not self.train_autoregression):
                trend_embedding_input = torch.cat((trend_init_enc, trend_output[:, self.seq_len: self.seq_len + i * self.stride, :], trend_zeros), dim = 1)
                trend_embedding = self.trend_embedding(trend_embedding_input)
                start = (((self.seq_conv_len + i - 1) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
                end = (((self.seq_conv_len + i) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
                trend_embedding = trend_embedding[: , start :end, :]
                trend_current_output = self.trend_decoder(trend_embedding, trend_embedding)
                trend_current_output = self.trend_trans_patch(trend_current_output)
                trend_output[:, self.seq_len + self.stride * i : self.seq_len + self.stride * (i + 1) , :] = trend_current_output 
            else:
                trend_embedding_input = torch.cat((trend_init_enc, trend_output[:, self.seq_len: self.seq_len + i * self.stride, :], trend_zeros), dim = 1)
                trend_embedding = self.trend_embedding(trend_embedding_input)
                start = (((self.seq_conv_len + i - 1) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
                end = (((self.seq_conv_len + i) *self.stride + 2 * self.patch_padding - self.patch_kernel) // self.patch_stride) + 1
                trend_embedding = trend_embedding[: , start :end, :]
                trend_current_output = self.trend_decoder(trend_embedding, trend_embedding)
                trend_current_output = self.trend_trans_patch(trend_current_output)
                trend_output[:, self.seq_len + self.stride * i : self.seq_len + self.stride * (i + 1) , :] = trend_current_output 
        trend_output = trend_output * \
                  (trend_stdev[:, 0, :].unsqueeze(1).repeat(1, self.actual_len, 1))
        trend_output = trend_output + \
                  (trend_means[:, 0, :].unsqueeze(1).repeat(1,  self.actual_len, 1))
        
        return seasonal_output + trend_output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -(self.actual_len - self.stride):, :]  # [B, L, D]
