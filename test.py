import torch

def beam_search_decoder(post, k):
    """Beam Search Decoder

    Parameters:

        post(Tensor) – the posterior of network.
        k(int) – beam size of decoder.

    Outputs:

        indices(Tensor) – a beam of index sequence.
        log_prob(Tensor) – a beam of log likelihood of sequence.

    Shape:

        post: (batch_size, seq_length, vocab_size).
        indices: (batch_size, beam_size, seq_length).
        log_prob: (batch_size, beam_size).

    Examples:

        >>> post = torch.softmax(torch.randn([32, 20, 1000]), -1)
        >>> indices, log_prob = beam_search_decoder(post, 3)

    """

    batch_size, seq_length, _ = post.shape
    log_post = post.log()
    log_prob, indices = log_post[:, 0, :].topk(k, dim = 1, sorted=True)
    indices = indices.unsqueeze(-1)
    for i in range(1, seq_length):
        log_prob = log_prob.unsqueeze(-1) + log_post[:, i, :].unsqueeze(1).repeat(1, k, 1)
        log_prob, index = log_prob.view(batch_size, -1).topk(k, sorted=True)
        indices = torch.cat([indices, index.unsqueeze(-1)], dim=-1)
    return indices, log_prob


def beam_search_decoder2(post, seq_val, seq_idx):
    """
        Post: (batch_size, k, vocab_size)
        seq_val: (batch_size, k, seq_length)
        seq_idx: (batch_size, k, seq_length)

    """

    batch_size, k, vocab_size = post.shape

    log_post = post.log()
    val, idx = log_post.topk(k, dim = 2)
    val = val.unsqueeze(-1)
    idx = idx.unsqueeze(-1)

    seq_val = seq_val.unsqueeze(2).repeat(1, 1, k, 1) # (batch_size, k, k, seq_length)
    seq_idx = seq_idx.unsqueeze(2).repeat(1, 1, k, 1)

    list_val = torch.cat((seq_val, val), -1).view(batch_size, k*k, -1)
    list_idx = torch.cat((seq_idx, idx), -1).view(batch_size, k*k, -1)

    _, rank_idx = list_val.sum(-1).topk(k) # (batch_size, k)

    dummy = rank_idx.unsqueeze(2).expand(rank_idx.size(0), rank_idx.size(1), list_val.size(2))
    seq_val = torch.gather(list_val, 1, dummy)
    seq_idx = torch.gather(list_idx, 1, dummy)

    return seq_val, seq_idx

if __name__ == '__main__':

    max_seqlen = 5
    vocab_size = 8
    batch = 4
    k = 3
    # post = torch.softmax(torch.randn([batch, 1, vocab_size]), -1)

    # for i in range(max_seqlen):
    #     new_post = torch.softmax(torch.randn([batch, 1, vocab_size]), -1)
    #     indices, log_prob = beam_search_decoder(post, k)
    #     post = torch.cat((post, new_post), 1)

    seq_val = torch.rand([batch, k, 1]).log()
    seq_idx = torch.ones([batch, k, 1])

    for i in range(max_seqlen):
        post = torch.softmax(torch.randn([batch, k, vocab_size]), -1)
        seq_val, seq_idx = beam_search_decoder2(post, seq_val, seq_idx)
    print(seq_idx)



    pass