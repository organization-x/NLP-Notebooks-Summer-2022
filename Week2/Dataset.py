class Dataset(Dataset):
  def __init__(self, X, Y, in_tknz, out_tknz, in_maxlen, out_maxlen):
    self.in_maxlen = in_maxlen
    self.out_maxlen = out_maxlen
    
    self.X = X # english sentences
    self.Y = Y # chinese sentences
    
    self.in_tknz = in_tknz # input tokenizer for english
    self.out_tknz = out_tknz # output tokenizer for chinese
  
  def __getitem__(self, idx):
    # -1 because we stil have to concate the <SOS> and <EOS> tokens
    enc_in = self.X[idx][:self.in_maxlen-1]
    enc_in = enc_in + ["<EOS>"]
    
    dec_in = self.Y[idx][:self.out_maxlen-1]
    dec_in = ["<SOS>"] + dec_in
    
    dec_out = self.Y[idx][:self.out_maxlen-1]
    dec_out = dec_out + ["<EOS>"]
    
    # Convert enc_in, dec_in, dec_out to 1D
    enc_in = self.in_tknz.transform(enc_in, max_len=self.in_maxlen, pad_first=False)
    dec_in = self.out_tknz.transform(dec_in, max_len=self.out_maxlen, pad_first=False)
    dec_out = self.out_tknz.transform(dec_out, max_len=self.out_maxlen, pad_first=False)

    return enc_in, dec_in, dec_out

  def __len__(self):
    # Returns number of data in this dataset
    return len(self.X)

# NOTE: collate_fn preprocesses your input from PyTorch Dataset above during PyTorch DataLoader
#       we can convert data into Long Tensors Here
def collate_fn(batch):
    '''
    param batch: ([enc_in, dec_in, dec_out]， [enc_in, dec_in, dec_out], output of getitem...)
    '''
    # unpack values
    enc_in, dec_in, dec_out = list(zip(*batch))
    # Return tensor type
    return torch.LongTensor(enc_in), torch.LongTensor(dec_in), torch.LongTensor(dec_out)

def get_dataloader(dataset, batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn):
    '''
    Returns a way to access and use the data
    '''
    dataloader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            drop_last=drop_last,
                            collate_fn=collate_fn)
    return dataloader