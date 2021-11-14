import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from GCN_lib.Rs_GCN import Rs_GCN, TextGCN, GCN

import opts
import misc.utils as utils
import torch.optim as optim

def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X

def l2norm2(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImagePrecompAttn(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False, if_IOU=True):
        super(EncoderImagePrecompAttn, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.if_IOU = if_IOU

        self.fc = nn.Linear(img_dim, embed_size)

        #self.weight_fc = nn.Linear(2, 1)
        self.img_rnn = nn.GRU(embed_size, embed_size, 1, batch_first=True)
        self.bn = nn.BatchNorm1d(embed_size) 

        self.init_weights()
        # GCN reasoning 
        if self.if_IOU:
          self.relationGCN = GCN(in_channels=embed_size, inter_channels=embed_size)
          self.relationGCN2 = GCN(in_channels=embed_size, inter_channels=embed_size)

        self.GCN = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.GCN2 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.GCN3 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)
        self.GCN4 = Rs_GCN(in_channels=embed_size, inter_channels=embed_size)



    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)


    def compute_pseudo(self, bbox, Iou):
        bb_size = (bbox[:, :, 2:] - bbox[:, :, :2])
        bb_centre = bbox[:, :, :2] + 0.5 * bb_size
        K = bb_centre.size(1)
        batch_size = bbox.size(0)

        # Compute cartesian coordinates (batch_size, K, K, 2)
        pseudo_coord = bb_centre.view(-1, K, 1, 2) - \
            bb_centre.view(-1, 1, K, 2)

        # Conver to polar coordinates
        rho = torch.sqrt(
            pseudo_coord[:, :, :, 0]**2 + pseudo_coord[:, :, :, 1]**2)

        return rho.cuda().float()


    def get_relation(self, query, context, smooth=9., eps=1e-8):
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

        # Get attention
        queryT = torch.transpose(query, 1, 2)
        attn = torch.bmm(context, queryT)
        query_norm = torch.norm(query,p=2,dim=2).repeat(1,queryL).view(batch_size_q, queryL, queryL).clamp(min=1e-8)
        source_norm = torch.norm(context,p=2,dim=2).repeat(1,sourceL).view(batch_size_q, sourceL, sourceL).clamp(min=1e-8)
        attn = torch.div(attn, query_norm)
        attn = torch.div(attn, source_norm)

        return attn

    def forward(self, images, bboxes, Iou):
    #def forward(self, images):
        """Extract image feature vectors."""
        fc_img_emd = self.fc(images)
        fc_img_emd = l2norm(fc_img_emd)
        
        GCN_img_emd0 = fc_img_emd.permute(0, 2, 1)
        GCN_img_emd = self.GCN(GCN_img_emd0)
        GCN_img_emd = self.GCN2(GCN_img_emd)
        GCN_img_emd = self.GCN3(GCN_img_emd)
        GCN_img_emd = self.GCN4(GCN_img_emd)
        GCN_img_emd = GCN_img_emd.permute(0, 2, 1)

        if self.if_IOU:
            weight = Iou.cuda().float()
            relation = self.get_relation(fc_img_emd, fc_img_emd)
            weight = weight.mul(relation)

            relationGCN = self.relationGCN(GCN_img_emd0, weight)
            relationGCN = self.relationGCN2(relationGCN, weight)
            relationGCN = relationGCN.permute(0, 2, 1)
            GCN_img_emd = (GCN_img_emd + relationGCN)/2
            GCN_img_emd = l2norm(GCN_img_emd)

        rnn_img, hidden_state = self.img_rnn(GCN_img_emd)

        # features = torch.mean(rnn_img,dim=1)
        features = hidden_state[0]

        features = self.bn(features) 
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)

        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)

        return features, GCN_img_emd

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecompAttn, self).load_state_dict(new_state)

# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, vocab_size, word_dim, embed_size, num_layers,
                 use_abs=False):
        super(EncoderText, self).__init__()
        self.use_abs = use_abs
        self.embed_size = embed_size

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        #self.embed = vocabs.Vectors(name='/home/cyh/cross-model/DARN/vocab/glove.840B.300d.txt')

        # caption embedding
        self.rnn = nn.GRU(word_dim, embed_size, num_layers, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(embed_size, embed_size, num_layers, batch_first=True)

        self.GCN = TextGCN(in_channels=embed_size, inter_channels=embed_size)
        self.init_weights()


    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def build_sparse_graph(self, depend, lens):
        totaladj = []
        temlen = max(lens)
        adj = np.zeros((len(lens), temlen, temlen))
        for j in range(len(depend)):
            dep = depend[j]
            for i, pair in enumerate(dep):
                if i == 0 or pair[0] >= temlen or pair[1] >= temlen:
                    continue
                adj[j, pair[0], pair[1]] = 1
                adj[j, pair[1], pair[0]] = 1
            adj[j] = adj[j] + np.eye(temlen)

        return torch.from_numpy(adj).cuda().float()

    
    def get_relation(self, query, context, smooth=9., eps=1e-8):
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

        # Get attention
        queryT = torch.transpose(query, 1, 2)
        attn = torch.bmm(context, queryT)
        query_norm = torch.norm(query,p=2,dim=2).repeat(1,queryL).view(batch_size_q, queryL, queryL).clamp(min=1e-8)
        source_norm = torch.norm(context,p=2,dim=2).repeat(1,sourceL).view(batch_size_q, sourceL, sourceL).clamp(min=1e-8)
        attn = torch.div(attn, query_norm)
        attn = torch.div(attn, source_norm)

        return attn

    def forward(self, x, lengths, depends):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded

        cap_emb = (cap_emb[:, :, :cap_emb.size(2) / 2] +
                       cap_emb[:, :, cap_emb.size(2) / 2:]) / 2

        adj_mtx = self.build_sparse_graph(depends, cap_len)
        relation = self.get_relation(cap_emb, cap_emb)

        adj_mtx = adj_mtx.mul(relation)
        GCN_cap_emd = cap_emb.permute(0, 2, 1)
        GCN_cap_emd = self.GCN(GCN_cap_emd, adj_mtx)
        GCN_cap_emd = GCN_cap_emd.permute(0, 2, 1)

        feature, _ = self.decoder(GCN_cap_emd)

        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        features = torch.gather(feature, 1, I).squeeze(1)
        
        # normalization in the joint embedding space
        features = l2norm(features)

        # take absolute value, used by order embeddings
        if self.use_abs:
            features = torch.abs(features)

        return features, cap_len, GCN_cap_emd

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param
        super(EncoderText, self).load_state_dict(new_state)



def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = False#max_violation

    def forward(self, im, s):
        # compute image-sentence score matrix
        #scores = self.sim(im, s)
        scores = self.sim(im,s)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)

        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()

class CGMN(object):

    def __init__(self, opt):
        # Build Models
        self.grad_clip = opt.grad_clip
        self.raw_feature_norm = opt.raw_feature_norm
        self.img_enc = EncoderImagePrecompAttn(opt.img_dim, opt.embed_size,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm,
                                    if_IOU = opt.if_IOU)
        self.txt_enc = EncoderText(opt.vocab_size, opt.word_dim,
                                   opt.embed_size, opt.num_layers,
                                   use_abs=opt.use_abs)

        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = False

        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.parameters())

        self.params = params

        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)

        self.Eiters = 0

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        """switch to train mode
        """
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        """switch to evaluate mode
        """
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, depend, bboxes, Iou, volatile=False):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        images = Variable(images)
        captions = Variable(captions)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        cap_emb, lens, GCN_cap_emb = self.txt_enc(captions, lengths, depend)
        img_emb, GCN_img_emb = self.img_enc(images, bboxes, Iou)

        return img_emb, cap_emb, GCN_img_emb, GCN_cap_emb

    def forward_loss(self, img_emb, cap_emb, **kwargs):
        """Compute the loss given pairs of image and caption embeddings
        """
        loss = self.criterion(img_emb, cap_emb)
        return loss

    def forward_matching_loss(self, captions, images, length):
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        n_region = images.size(1)

        for i in range(n_caption):
            cap_i_expand = captions[i,:length[i],:].repeat(n_image, 1, 1)
            row_sim = self.get_relation(images, cap_i_expand)
            row_sim = row_sim.max(dim=2, keepdim=True)[0]
            row_sim = torch.sum(row_sim, dim=1)
            similarities.append(row_sim)

        scores = self.getscore(similarities, n_caption)
        return scores


    def getscore(self, similarities, n_image):
        scores = torch.cat(similarities, 1)

        diagonal = scores.diag().view(n_image, 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (0.2 + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (0.2 + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        return cost_s.sum() + cost_im.sum()

    def get_relation(self, query, context, smooth=9., eps=1e-8):
        batch_size_q, queryL = query.size(0), query.size(1)
        batch_size, sourceL = context.size(0), context.size(1)

        # Get attention
        queryT = torch.transpose(query, 1, 2)
        attn = torch.bmm(context, queryT)
        query_norm = torch.norm(query,p=2,dim=2).repeat(1,sourceL).view(batch_size_q, sourceL, queryL).clamp(min=eps)
        source_norm = torch.norm(context,p=2,dim=2).repeat(1,queryL).view(batch_size_q, queryL, sourceL).clamp(min=eps)
        source_norm = torch.transpose(source_norm, 1, 2)
        attn = torch.div(attn, query_norm)
        attn = torch.div(attn, source_norm)

        return attn

    def train_emb(self, images, captions, lengths, ids, depend, bboxes, Iou, caption_labels, caption_masks, *args):
        """One training step given images and captions.
        """
        self.Eiters += 1
        self.logger.update('Eit', self.Eiters)
        self.logger.update('lr', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, GCN_img_emd, GCN_cap_emd = self.forward_emb(images, captions, lengths, depend, bboxes, Iou)

        self.optimizer.zero_grad()
        retrieval_loss = self.forward_loss(img_emb, cap_emb)

        self.optimizer.zero_grad()
        graph_matching_loss = self.forward_matching_loss(GCN_cap_emd, GCN_img_emd, lengths)
        loss = retrieval_loss + graph_matching_loss/3

        # compute gradient and do SGD step
        loss.backward()

        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()