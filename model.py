import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class LinearUnit(nn.Module):
    def __init__(self, in_features, out_features, batchnorm=True, nonlinearity=nn.LeakyReLU(0.2)):
        super(LinearUnit, self).__init__()
        if batchnorm is True:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features),
                    nn.BatchNorm1d(out_features), nonlinearity)
        else:
            self.model = nn.Sequential(
                    nn.Linear(in_features, out_features), nonlinearity)
    
    def forward(self, x):
        return self.model(x)


class conv(nn.Module):
    def __init__(self, nin, nout):
        super(conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )
    
    def forward(self, input):
        return self.main(input)


class upconv(nn.Module):
    def __init__(self, nin, nout):
        super(upconv, self).__init__()
        self.main = nn.Sequential(
                nn.ConvTranspose2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )
    
    def forward(self, input):
        return self.main(input)


class encoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(encoder, self).__init__()
        self.dim = dim
        nf = 64
        self.c1 = conv(nc, nf)
        self.c2 = conv(nf, nf * 2)
        self.c3 = conv(nf * 2, nf * 4)
        self.c4 = conv(nf * 4, nf * 8)
        self.c5 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 4, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )

    def forward(self, input):
        h1 = self.c1(input)
        h2 = self.c2(h1)
        h3 = self.c3(h2)
        h4 = self.c4(h3)
        h5 = self.c5(h4)
        return h5.view(-1, self.dim), None


class decoder(nn.Module):
    def __init__(self, dim, nc=1):
        super(decoder, self).__init__()
        self.dim = dim
        nf = 64
        self.upc1 = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(dim, nf * 8, 4, 1, 0),
                nn.BatchNorm2d(nf * 8),
                nn.LeakyReLU(0.2, inplace=True)
                )
        self.upc2 = upconv(nf * 8, nf * 4)
        self.upc3 = upconv(nf * 4, nf * 2)
        self.upc4 = upconv(nf * 2, nf)
        self.upc5 = nn.Sequential(
                nn.ConvTranspose2d(nf, nc, 4, 2, 1),
                nn.Sigmoid()
                )

    def forward(self, input):
        d1 = self.upc1(input.view(-1, self.dim, 1, 1))
        d2 = self.upc2(d1)
        d3 = self.upc3(d2)
        d4 = self.upc4(d3)
        output = self.upc5(d4)
        output = output.view(input.shape[0], input.shape[1], output.shape[1], output.shape[2], output.shape[3])

        return output

class classifier_Sprite_all(nn.Module):
    def __init__(self, opt):
        super(classifier_Sprite_all, self).__init__()
        self.g_dim = opt.g_dim  # frame feature
        self.channels = opt.channels  # frame feature
        self.hidden_dim = opt.rnn_size
        self.frames = opt.frames
        self.encoder = encoder(self.g_dim, self.channels)
        self.bilstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.cls_skin = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_top = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_pant = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_hair = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 6))
        self.cls_action = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, 9))

    def encoder_frame(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def forward(self, x):
        conv_x = self.encoder_frame(x)
        lstm_out, _ = self.bilstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        return self.cls_action(lstm_out_f), self.cls_skin(lstm_out_f), self.cls_pant(lstm_out_f), \
               self.cls_top(lstm_out_f), self.cls_hair(lstm_out_f)


class CDSVAE(nn.Module):
    def __init__(self, opt):
        super(CDSVAE, self).__init__()
        self.f_dim = opt.f_dim  # content
        self.z_dim = opt.z_dim  # motion
        self.g_dim = opt.g_dim 
        self.channels = opt.channels 
        self.hidden_dim = opt.rnn_size
        self.f_rnn_layers = opt.f_rnn_layers
        self.frames = opt.frames

        self.encoder = encoder(self.g_dim, self.channels)
        self.decoder = decoder(self.z_dim + self.f_dim, self.channels)

        self.z_prior_lstm_ly1 = nn.LSTMCell(self.z_dim, self.hidden_dim)
        self.z_prior_lstm_ly2 = nn.LSTMCell(self.hidden_dim, self.hidden_dim)

        self.z_prior_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_prior_logvar = nn.Linear(self.hidden_dim, self.z_dim)

        self.z_lstm = nn.LSTM(self.g_dim, self.hidden_dim, 1, bidirectional=True, batch_first=True)
        self.f_mean = LinearUnit(self.hidden_dim * 2, self.f_dim, False)
        self.f_logvar = LinearUnit(self.hidden_dim * 2, self.f_dim, False)

        self.z_rnn = nn.RNN(self.hidden_dim * 2, self.hidden_dim, batch_first=True)
        
        self.z_mean = nn.Linear(self.hidden_dim, self.z_dim)
        self.z_logvar = nn.Linear(self.hidden_dim, self.z_dim)

    def encode_and_sample_post(self, x):
        if isinstance(x, list):
            conv_x = self.encoder_frame(x[0])
        else:
            conv_x = self.encoder_frame(x)
        lstm_out, _ = self.z_lstm(conv_x)
        backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
        frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
        lstm_out_f = torch.cat((frontal, backward), dim=1)
        f_mean = self.f_mean(lstm_out_f)
        f_logvar = self.f_logvar(lstm_out_f)
        f_post = self.reparameterize(f_mean, f_logvar, random_sampling=True)

        features, _ = self.z_rnn(lstm_out)
        z_mean = self.z_mean(features)
        z_logvar = self.z_logvar(features)
        z_post = self.reparameterize(z_mean, z_logvar, random_sampling=True)

        if isinstance(x, list):
            f_mean_list = [f_mean]
            for _x in x[1:]:
                conv_x = self.encoder_frame(_x)
                lstm_out, _ = self.z_lstm(conv_x)
                backward = lstm_out[:, 0, self.hidden_dim:2 * self.hidden_dim]
                frontal = lstm_out[:, self.frames - 1, 0:self.hidden_dim]
                lstm_out_f = torch.cat((frontal, backward), dim=1)
                f_mean = self.f_mean(lstm_out_f)
                f_mean_list.append(f_mean)
            f_mean = f_mean_list
        
        return f_mean, f_logvar, f_post, z_mean, z_logvar, z_post

    def forward(self, x):
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)
        z_mean_prior, z_logvar_prior, z_prior = self.sample_z_prior_train(x.size(0), random_sampling=self.training)
        
        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, z_prior, \
               recon_x

    def forward_fixed_motion(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        z_repeat = z_post[0].repeat(z_post.shape[0], 1, 1)
        f_expand = f_post.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_repeat, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_content(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=self.training)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_repeat = f_post[0].repeat(f_post.shape[0], 1)
        f_expand = f_repeat.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_post, f_expand), dim=2)
        recon_x = self.decoder(zf)
        return f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post, z_mean_prior, z_logvar_prior, recon_x

    def forward_fixed_content_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)

        zf = torch.cat((z_mean_prior, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def forward_fixed_motion_for_classification(self, x):
        z_mean_prior, z_logvar_prior, _ = self.sample_z(x.size(0), random_sampling=True)
        f_mean, f_logvar, f_post, z_mean_post, z_logvar_post, z_post = self.encode_and_sample_post(x)

        f_prior = self.reparameterize(torch.zeros(f_mean.shape).cuda(), torch.zeros(f_logvar.shape).cuda(),
                                        random_sampling=True)
        f_expand = f_prior.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x_sample = self.decoder(zf)

        f_expand = f_mean.unsqueeze(1).expand(-1, self.frames, self.f_dim)
        zf = torch.cat((z_mean_post, f_expand), dim=2)
        recon_x = self.decoder(zf)

        return recon_x_sample, recon_x

    def encoder_frame(self, x):
        x_shape = x.shape
        x = x.view(-1, x_shape[-3], x_shape[-2], x_shape[-1])
        x_embed = self.encoder(x)[0]
        return x_embed.view(x_shape[0], x_shape[1], -1)

    def reparameterize(self, mean, logvar, random_sampling=True):
        if random_sampling is True:
            eps = torch.randn_like(logvar)
            std = torch.exp(0.5 * logvar)
            z = mean + eps * std
            return z
        else:
            return mean

    def sample_z_prior_test(self, n_sample, n_frame, random_sampling=True):
        z_out = None 
        z_means = None
        z_logvars = None
        batch_size = n_sample

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(n_frame):
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_prior
        return z_means, z_logvars, z_out

    def sample_z_prior_train(self, batch_size, random_sampling=True):
        z_out = None 
        z_means = None
        z_logvars = None

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()

        for i in range(self.frames):
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_prior = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_prior.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_prior.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
            z_t = z_prior
        return z_means, z_logvars, z_out

    def sample_z(self, batch_size, random_sampling=True):
        z_out = None 
        z_means = None
        z_logvars = None

        z_t = torch.zeros(batch_size, self.z_dim).cuda()
        h_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly1 = torch.zeros(batch_size, self.hidden_dim).cuda()
        h_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        c_t_ly2 = torch.zeros(batch_size, self.hidden_dim).cuda()
        for _ in range(self.frames):
            h_t_ly1, c_t_ly1 = self.z_prior_lstm_ly1(z_t, (h_t_ly1, c_t_ly1))
            h_t_ly2, c_t_ly2 = self.z_prior_lstm_ly2(h_t_ly1, (h_t_ly2, c_t_ly2))

            z_mean_t = self.z_prior_mean(h_t_ly2)
            z_logvar_t = self.z_prior_logvar(h_t_ly2)
            z_t = self.reparameterize(z_mean_t, z_logvar_t, random_sampling)
            if z_out is None:
                z_out = z_t.unsqueeze(1)
                z_means = z_mean_t.unsqueeze(1)
                z_logvars = z_logvar_t.unsqueeze(1)
            else:
                z_out = torch.cat((z_out, z_t.unsqueeze(1)), dim=1)
                z_means = torch.cat((z_means, z_mean_t.unsqueeze(1)), dim=1)
                z_logvars = torch.cat((z_logvars, z_logvar_t.unsqueeze(1)), dim=1)
        return z_means, z_logvars, z_out

