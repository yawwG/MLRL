import torch
import torch.nn as nn

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

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
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)
meseloss_func = nn.MSELoss(reduction='mean').cuda()
def g_loss(cnn_code1, cnn_code2, rnn_code1, rnn_code2, eps=1e-8, temp3=10.0):

    # batch_size = cnn_code1.shape[0]
    # labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code1.device)
    # labels = torch.zeros(batch_size,dtype=torch.long).cuda()

    if cnn_code1.dim() == 2:
        cnn_code1 = cnn_code1.unsqueeze(0)
        rnn_code1 = rnn_code1.unsqueeze(0)
        cnn_code2 = cnn_code2.unsqueeze(0)
        rnn_code2 = rnn_code2.unsqueeze(0)

    cnn_code_norm1 = torch.norm(cnn_code1, 2, dim=2, keepdim=True)
    rnn_code_norm1 = torch.norm(rnn_code1, 2, dim=2, keepdim=True)
    cnn_code_norm2 = torch.norm(cnn_code2, 2, dim=2, keepdim=True)
    rnn_code_norm2 = torch.norm(rnn_code2, 2, dim=2, keepdim=True)

    scores0_1 = torch.bmm(cnn_code1, rnn_code1.transpose(1, 2))
    norm0_1 = torch.bmm(cnn_code_norm1, rnn_code_norm1.transpose(1, 2))
    scores0_1 = scores0_1 / norm0_1.clamp(min=eps) * temp3

    scores0_2 = torch.bmm(cnn_code2, rnn_code2.transpose(1, 2))
    norm0_2 = torch.bmm(cnn_code_norm2, rnn_code_norm2.transpose(1, 2))
    scores0_2 = scores0_2 / norm0_2.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0_1 = scores0_1.view(scores0_1.size(1),scores0_1.size(2))
    scores1_1 = scores0_1.transpose(0, 1)
    scores0_2 = scores0_2.view(scores0_2.size(1),scores0_2.size(2))
    scores1_2 = scores0_2.transpose(0, 1)

    # loss0 =  cosine_similarity(scores0_1, scores0_2)
    # loss1 = cosine_similarity(scores1_1, scores1_2)
    loss0 = meseloss_func(scores0_1, scores0_2)
    loss1 = meseloss_func(scores1_1, scores1_2)
    return loss0+loss1

def cal_l_loss(img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"):
    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 5]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 5]
        word = word.repeat(batch_size, 1, 1)  # [8, 768, 5]
        context = img_features  # [8, 768, 19, 19]

        weiContext, attn = attention_fn1(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [40, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [40, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [8, 5]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [8, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [8, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [8, 8]
    return similarities, similarities1, att_maps

def l_loss(
    img_features2, words_emb2, img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):
    similarities, simlarities_, attention_maps_p = cal_l_loss(
       img_features, words_emb, cap_lens
    )
    similarities2, simlarities2_, attention_maps_c = cal_l_loss(
       img_features2, words_emb2, cap_lens
    )

    loss0 = meseloss_func(similarities, similarities2)  # labels: arange(batch_size)
    loss1 = meseloss_func(simlarities_, simlarities2_)
    # return loss0, loss1, att_maps, attn1
    return loss0+loss1, attention_maps_p, attention_maps_c

def lva_lta_l_loss(
    img_features2, img_features, words_emb2, words_emb, cap_lens2, cap_lens1, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):
    similarities, simlarities_, attention_maps_lva = cal_l_loss(
        img_features, img_features2, cap_lens2
    )
    similarities2, simlarities2_, attention_maps_lta = cal_l_loss(
        words_emb, words_emb2, cap_lens1
    )

    loss0 = meseloss_func(similarities, similarities2)  # labels: arange(batch_size)
    loss1 = meseloss_func(simlarities_, simlarities2_)
    # return loss0, loss1, att_maps, attn1
    return loss0 + loss1, attention_maps_lva, attention_maps_lta