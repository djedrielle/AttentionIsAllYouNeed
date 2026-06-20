import torch

def scaled_dot_product_attention(Q, K, V, mask=None):
    
    mScores = torch.matmul(Q, K.transpose(-2, -1))
    if mask is not None:
        mScores = mScores.masked_fill(mask == 0, float('-inf'))
    mPesos = torch.softmax(mScores/(Q.size(-1) ** 0.5), dim=-1)
    mValueResultante = torch.matmul(mPesos, V)

    return mValueResultante, mPesos