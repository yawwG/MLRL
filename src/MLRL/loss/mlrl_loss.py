import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / ((((w1 * w2) + + 1**(-12)).clamp(min=eps)) + 1**(-12)) ).squeeze()

def attention_fn1(query, context, temp1):
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw
    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()
    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    if (torch.isinf(contextT).any()):
        print('')
    if (torch.isinf(query).any()):
        print('')
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn+(1**(-12)))
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)
    return weightedContext, attn.view(batch_size, -1, ih, iw)
#----------------------
def rec_loss(rec, images):
    mse_loss = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
    rec_loss = mse_loss( rec, images) 

    return rec_loss
#----------------------
def inner_loss_l(cnn_code, rnn_mlp1, rnn_mlp2, b, eps=1e-8, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"):#rnn[2,768,2]

    batch_size = cnn_code.shape[0]
    i = 0
    cc = []
    mlo = []
    while i < batch_size:
        cc.append(torch.split(cnn_code,1,dim=0)[i])
        mlo.append(torch.split(cnn_code, 1, dim=0)[i+1])
        i = i+2
    cc = torch.stack(cc)
    cc = torch.squeeze(cc, dim=1)
    mlo = torch.stack(mlo)
    mlo = torch.squeeze(mlo, dim=1)
    # cc = torch.cat((torch.split(cnn_code,1,dim=0)[0], torch.split(cnn_code,1,dim=0)[2], torch.split(cnn_code,1,dim=0)[4],torch.split(cnn_code,1,dim=0)[6],torch.split(cnn_code,1,dim=0)[8],torch.split(cnn_code,1,dim=0)[10]),dim=0)
    # mlo = torch.cat((torch.split(cnn_code,1,dim=0)[1], torch.split(cnn_code,1,dim=0)[3], torch.split(cnn_code,1,dim=0)[5],torch.split(cnn_code,1,dim=0)[7],torch.split(cnn_code,1,dim=0)[9],torch.split(cnn_code,1,dim=0)[11]),dim=0)
    mlo = mlo.view(mlo.size(0), mlo.size(1), mlo.size(2) * mlo.size(3))
    img_features = cc
    words_emb = mlo
    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = words_emb.size(2)  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        # weiContext, attn, adv_loss, flag = attention_fn(
        #     word, context, temp1, label
        # )  # [48, 768, 25], [48, 25, 19, 19]
        weiContext, attn = attention_fn1(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
        # ifnan1 = torch.isnan(word).any()
        # ifnan2 = torch.isnan(weiContext).any()
        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]

        row_sim = torch.log(row_sim)
        # if (torch.isnan(row_sim).any()):
        #     continue
        # else:
        similarities.append(row_sim)
    # try:
    similarities = torch.cat(similarities, 1)  #

    similarities = similarities * temp3

    labels_inner = Variable(torch.LongTensor(range(similarities.size(0)))).cuda()  # [0,1]

    similarities1_ = similarities.transpose(0, 1)

    loss_inner = nn.CrossEntropyLoss()(similarities , labels_inner )  # labels: arange(batch_size)
    loss_inner1 = nn.CrossEntropyLoss()(similarities1_ , labels_inner)

    return loss_inner+loss_inner1

def inner_loss_g(cnn_code, rnn_mlp1, rnn_mlp2, b,  eps=1e-8, temp3=10.0):
    batch_size = cnn_code.shape[0]
    i = 0
    cc = []
    mlo = []
    while i < batch_size:
        cc.append(torch.split(cnn_code, 1, dim=0)[i])
        mlo.append(torch.split(cnn_code, 1, dim=0)[i + 1])
        i = i + 2
    cc = torch.stack(cc)
    cc = torch.squeeze(cc, dim=1)
    mlo = torch.stack(mlo)
    mlo = torch.squeeze(mlo, dim=1)

    if cc.dim() == 2:
        cc = cc.unsqueeze(0)
        mlo = mlo.unsqueeze(0)
        # rnn_mlp1 = rnn_mlp1.unsqueeze(0)
        # rnn_mlp2 = rnn_mlp2.unsqueeze(0)

    cc_norm = torch.norm(cc, 2, dim=2, keepdim=True)
    mlo_norm = torch.norm(mlo, 2, dim=2, keepdim=True)

    # rnn_mlp1_norm = torch.norm(rnn_mlp1, 2, dim=2, keepdim=True)
    # rnn_mlp2_norm = torch.norm(rnn_mlp2, 2, dim=2, keepdim=True)

    labels = Variable(torch.LongTensor(range(int(batch_size / 2)))).to(cnn_code.device)
    scores0 = torch.bmm(cc, mlo.transpose(1, 2))
    norm0 = torch.bmm(cc_norm, mlo_norm.transpose(1, 2))
    scores0 = (scores0 / (norm0.clamp(min=eps) * temp3)) +1**(-12)
    #
    # scores0_rnn = torch.bmm(rnn_mlp1, rnn_mlp2.transpose(1, 2))
    # norm0_rnn = torch.bmm(rnn_mlp1_norm, rnn_mlp2_norm.transpose(1, 2))
    # scores0_rnn = scores0_rnn / norm0_rnn.clamp(min=eps) * temp3
    # torch.nan_to_num(scores0)
    # --> batch_size x batch_size   inner
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)


    loss0 = nn.CrossEntropyLoss()(scores0,  labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)

    # scores0_rnn = scores0_rnn.squeeze()
    # scores1_rnn = scores0_rnn.transpose(0, 1)
    # rnn_labels = Variable(torch.LongTensor(range(int(batch_size / 4)))).to(cnn_code.device)
    # loss0_rnn = nn.CrossEntropyLoss()(scores0_rnn, rnn_labels)
    # loss1_rnn = nn.CrossEntropyLoss()(scores1_rnn, rnn_labels)

    return loss0 + loss1

def inner_loss_side_g(cc, mlo, b, eps=1e-8, temp3=10.0):
    batch_size = cc.shape[0]
    i = 0
    # cc = []
    # mlo = []
    # while i < batch_size:
    #     cc.append(torch.split(cnn_code, 1, dim=0)[i])
    #     mlo.append(torch.split(cnn_code, 1, dim=0)[i + 1])
    #     i = i + 2
    # cc = torch.stack(cc)
    # cc = torch.squeeze(cc, dim=1)
    # mlo = torch.stack(mlo)
    # mlo = torch.squeeze(mlo, dim=1)


    if cc.dim() == 2:
        cc = cc.unsqueeze(0)
        mlo = mlo.unsqueeze(0)
        # rnn_mlp1 = rnn_mlp1.unsqueeze(0)
        # rnn_mlp2 = rnn_mlp2.unsqueeze(0)

    cc_norm = torch.norm(cc, 2, dim=2, keepdim=True)
    mlo_norm = torch.norm(mlo, 2, dim=2, keepdim=True)

    # rnn_mlp1_norm = torch.norm(rnn_mlp1, 2, dim=2, keepdim=True)
    # rnn_mlp2_norm = torch.norm(rnn_mlp2, 2, dim=2, keepdim=True)

    labels = Variable(torch.LongTensor(range(int(batch_size)))).to(cc.device)
    scores0 = torch.bmm(cc, mlo.transpose(1, 2))
    norm0 = torch.bmm(cc_norm, mlo_norm.transpose(1, 2))
    scores0 = (scores0 / (norm0.clamp(min=eps) * temp3)) +1**(-12)
    #
    # scores0_rnn = torch.bmm(rnn_mlp1, rnn_mlp2.transpose(1, 2))
    # norm0_rnn = torch.bmm(rnn_mlp1_norm, rnn_mlp2_norm.transpose(1, 2))
    # scores0_rnn = scores0_rnn / norm0_rnn.clamp(min=eps) * temp3

    # --> batch_size x batch_size   inner
    scores0 = scores0.squeeze()
    # torch.nan_to_num(scores0)
    scores1 = scores0.transpose(0, 1)


    loss0 = nn.CrossEntropyLoss()(scores0,  labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)

    # scores0_rnn = scores0_rnn.squeeze()
    # scores1_rnn = scores0_rnn.transpose(0, 1)
    # rnn_labels = Variable(torch.LongTensor(range(int(batch_size / 4)))).to(cnn_code.device)
    # loss0_rnn = nn.CrossEntropyLoss()(scores0_rnn, rnn_labels)
    # loss1_rnn = nn.CrossEntropyLoss()(scores1_rnn, rnn_labels)

    return loss0 + loss1
#------------------------------------------------------------------
def global_loss(cnn_code, rnn_code, keyword, flag, eps=1e-8, temp3=10.0):

    batch_size = cnn_code.shape[0]
    # labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)
    labels_inner = Variable(torch.LongTensor(range(int(batch_size/4)))).cuda()  # [0,1]
    # labels = torch.zeros(batch_size,dtype=torch.long).cuda()

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = (scores0 / (norm0.clamp(min=eps) * temp3)) +1**(-12)

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    # scores1 = scores0.transpose(0, 1)
    similarities_ = []

    n = 0
    for i in range(int(batch_size / 4)):
        j = 0
        while ((n + j + 1) * 4) <= batch_size:
            similarities_.append(torch.mean(scores0[((n + i) * 4):((n + i + 1) * 4), ((n + j) * 4):((n + j + 1) * 4)]))  # [0:4,0:4]
            j = j+1
    # for i in range(int(batch_size/4)):
    #     similarities_.append(torch.mean(scores0[((n+i) * 4):((n+i+1) * 4), (n * 4):((n + 1) * 4)]))  # [0:4,0:4]
    #     similarities_.append(torch.mean(scores0[((n+i) * 4):((n+i+1) * 4), ((n + 1) * 4):(n + 2) * 4]))  # [0:4,4:8]
    #     similarities_.append(torch.mean(scores0[((n+i) * 4):((n+i+1) * 4), ((n + 2) * 4):(n + 3) * 4]))  # [0:4,8:12]

        # similarities_.append(torch.mean(scores0[((n + 1) * 4):(n + 2) * 4, (n * 4):((n + 1) * 4)]))  # [4:8,0:4]
        # similarities_.append(torch.mean(scores0[((n + 1) * 4):(n + 2) * 4, ((n + 1) * 4):((n + 2) * 4)]))  # [4:8,4:8]
        # similarities_.append(torch.mean(scores0[((n + 1) * 4):(n + 2) * 4, ((n + 2) * 4):((n + 3) * 4)]))  # [4:8,8:12]
        #
        # similarities_.append(torch.mean(scores0[((n + 2) * 4):(n + 3) * 4, (n * 4):((n + 1) * 4)]))  # [4:8,0:4]
        # similarities_.append(torch.mean(scores0[((n + 2) * 4):(n + 3) * 4, ((n + 1) * 4):((n + 2) * 4)]))  # [4:8,4:8]
        # similarities_.append(torch.mean(scores0[((n + 2) * 4):(n + 3) * 4, ((n + 2) * 4):((n + 3) * 4)]))  # [4:8,8:12]

    similarities_ = torch.stack(similarities_).view(int(batch_size/4), int(batch_size/4))
    # torch.nan_to_num(similarities_)
    similarities1_ = similarities_.transpose(0, 1)

    i = 0
    flag_ = []
    while i < batch_size:
        flag_.append(torch.split(flag, 1, dim=0)[i])
        i = i + 4
    flag = torch.stack(flag_)
    flag = torch.squeeze(flag, dim=1)
    # flag = torch.cat((torch.split(flag, 1, dim=0)[0],
    #                   torch.split(flag, 1, dim=0)[4],
    #                   torch.split(flag, 1, dim=0)[8]), dim=0)

    loss0 = nn.CrossEntropyLoss()(similarities_ * flag, labels_inner * flag)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1_ * flag, labels_inner * flag)
    # #coarse-to-fine loss
    # idx_rank, rank = rank_kekyword(keyword, similarities_)
    # label_ranks = []
    #
    # for i in range(len(rank) - 1):
    #     t = i + 1
    #     for j in range(t, len(rank)):
    #         for k in range(len(rank[i][0])):
    #
    #             for w in range(len(rank[j][0])):
    #                 a = rank[i][0]
    #                 label_ranks.append(a[k])
    #                 b = rank[j][0]
    #                 label_ranks.append(b[w])
    #         if (t < len(rank)):
    #             t += 1
    #
    # if (len(label_ranks) == 0):
    #     coarse_to_fine_loss = 0
    # else:
    #     a = int(len(label_ranks) / 2)
    #     label_ranks = torch.stack(label_ranks).view(a, 2)
    #     label_rank = torch.ones(a, dtype=torch.long).cuda()
    #     coarse_to_fine_loss = nn.CrossEntropyLoss()(label_ranks, label_rank)

    # return loss0, loss1, coarse_to_fine_loss
    return loss0, loss1, 0


def local_loss(
    img_features, words_emb, cap_lens, label, keyword, flag, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):

    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        # weiContext, attn, adv_loss, flag = attention_fn(
        #     word, context, temp1, label
        # )  # [48, 768, 25], [48, 25, 19, 19]
        weiContext, attn = attention_fn1(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim + 1**(-12))

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    # torch.nan_to_num(similarities)
    similarities = similarities * temp3


    # ****************inner
    # similarities = torch.mean(similarities[])
    labels_inner = Variable(torch.LongTensor(range(int(batch_size/4)))).cuda() # [0,1]
    similarities_ = []
    n = 0
    j = 0
    for i in range(int(batch_size / 4)):
        j = 0
        while ((n + j + 1) * 4) <= batch_size:
            similarities_.append(torch.mean(similarities[((n + i) * 4):((n + i + 1) * 4), ((n + j) * 4):((n + j + 1) * 4)]))  # [0:4,0:4]
            j = j+1
        # continue
            # similarities_.append(
            #     torch.mean(similarities[((n + i) * 4):((n + i + 1) * 4), ((n + 1) * 4):(n + 2) * 4]))  # [0:4,4:8]
            # similarities_.append(
            #     torch.mean(similarities[((n + i) * 4):((n + i + 1) * 4), ((n + 2) * 4):(n + 3) * 4]))  # [0:4,8:12]
    # for n in range(1):
    #     similarities_.append(torch.mean(similarities[(n * 4):((n + 1) * 4), (n * 4):((n + 1) * 4)]))  # [0:4,0:4]
    #     similarities_.append(torch.mean(similarities[(n * 4):((n + 1) * 4), ((n + 1) * 4):(n + 2) * 4]))  # [0:4,4:8]
    #     similarities_.append(torch.mean(similarities[(n * 4):((n + 1) * 4), ((n + 2) * 4):(n + 3) * 4]))  # [0:4,8:12]
    #     similarities_.append(torch.mean(similarities[((n + 1) * 4):(n + 2) * 4, (n * 4):((n + 1) * 4)]))  # [4:8,0:4]
    #     similarities_.append(
    #         torch.mean(similarities[((n + 1) * 4):(n + 2) * 4, ((n + 1) * 4):((n + 2) * 4)]))  # [4:8,4:8]
    #     similarities_.append(
    #         torch.mean(similarities[((n + 1) * 4):(n + 2) * 4, ((n + 2) * 4):((n + 3) * 4)]))  # [4:8,8:12]
    #
    #     similarities_.append(torch.mean(similarities[((n + 2) * 4):(n + 3) * 4, (n * 4):((n + 1) * 4)]))  # [4:8,0:4]
    #     similarities_.append(
    #         torch.mean(similarities[((n + 2) * 4):(n + 3) * 4, ((n + 1) * 4):((n + 2) * 4)]))  # [4:8,4:8]
    #     similarities_.append(
    #         torch.mean(similarities[((n + 2) * 4):(n + 3) * 4, ((n + 2) * 4):((n + 3) * 4)]))  # [4:8,8:12]
    # labels_inner = torch.stack([itm for itm in labels_inner for i in range(2)])  # [0,0,1,1,2,2,3,3,4,4,5,5]
    similarities_ = torch.stack(similarities_).view(int(batch_size / 4), int(batch_size / 4))
    similarities1_ = similarities_.transpose(0, 1)
    i = 0
    flag_ = []
    while i < batch_size:
        flag_.append(torch.split(flag, 1, dim=0)[i])
        i = i + 4
    flag = torch.stack(flag_)
    flag = torch.squeeze(flag, dim=1)

    # flag = torch.cat((torch.split(flag, 1, dim=0)[0],
    #                   torch.split(flag, 1, dim=0)[4],
    #                   torch.split(flag, 1, dim=0)[8]), dim=0)
    loss_inner = nn.CrossEntropyLoss()(similarities_*flag, labels_inner*flag)  # labels: arange(batch_size)
    loss_inner1 = nn.CrossEntropyLoss()(similarities1_*flag, labels_inner*flag)


    # # coarse-to-fine loss
    # idx_rank, rank =  rank_kekyword(keyword, similarities_)
    # label_ranks = []
    #
    # for i in range(len(rank) - 1):
    #     t = i + 1
    #     for j in range(t, len(rank)):
    #         for k in range(len(rank[i][0])):
    #             for w in range(len(rank[j][0])):
    #                 a = rank[i][0]
    #                 label_ranks.append(a[k])
    #                 b = rank[j][0]
    #                 label_ranks.append(b[w])
    #         if (t < len(rank)):
    #             t += 1
    #
    # if(len(label_ranks)==0):
    #     coarse_to_fine_loss = 0
    # else:
    #     a = int(len(label_ranks) / 2)
    #     label_ranks = torch.stack(label_ranks).view(a, 2)
    #     label_rank = torch.ones(a,dtype=torch.long).cuda()
    #     coarse_to_fine_loss = nn.CrossEntropyLoss()(label_ranks, label_rank)

    return loss_inner, loss_inner1, att_maps, 0
    # return loss_inner, loss_inner1, att_maps, coarse_to_fine_loss

def rank_kekyword(keyword, similarities):
    nodumplicate_Keyword =keyword

    rank = []
    for i in range(int(len(nodumplicate_Keyword))):
        for j in range(int(len(nodumplicate_Keyword))):
            if (i != j) and (i < j):
               ranktmp=0
               # aa = nodumplicate_Keyword[i].split(' ')
               for key in nodumplicate_Keyword[i].split(' '):
                   if key in nodumplicate_Keyword[j].split(' ') and key!='[CLS]' and key!='[SEP]' and key!='[PAD]' and key!='':
                       ranktmp +=1
               rank.append(ranktmp)
    intra = []
    for i in range(similarities.size(0)):
        for j in range(similarities.size(1)):
            if (i != j) and (i < j):
                intra.append((similarities[i, j] + similarities[j, i]) / 2)
    inta_ranked =[]
    rankindex = np.argsort(rank)
    for i in range(len(intra)):
        inta_ranked.append(intra[rankindex[i]])

    k = 0
    s1 = list(inta_ranked)
    for j in range(len(rankindex) - 1):
        if rank[rankindex[j]] != rank[rankindex[j + 1]]:
            k += 1
            s1.insert(j + k, '#')
    j=0
    s2=[]
    for i in range(len(s1)):
        if (i == len(s1) - 1):
            s2.append([s1[j:(i+1)]])
        if str(s1[i]) =='#':
            s2.append([s1[j:(i)]])
            j=i+1
    return rankindex, s2