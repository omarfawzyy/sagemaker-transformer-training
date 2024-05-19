import sentencepiece as sp
import torch
import io
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

def count_examples(file,sp_model=None,max_size=None):
  i = 0
  for ex in iter(io.open(file, encoding="utf8")):
    if sp_model is not None and max_size is not None:
      if len(sp_model.encode(ex)<=max_size:
        i+=1
    else:
      i+=1
  return i

def create_data_chunk(src_file,trg_file,sp_model,max_len,idx_slice):
  raw_src_iter = iter(io.open(src_file, encoding="utf8"))
  raw_trg_iter = iter(io.open(trg_file, encoding="utf8"))
  min_idx,max_idx = idx_slice
  data = []
  for i,(raw_src, raw_trg) in enumerate(zip(raw_src_iter, raw_trg_iter)):
    if i < min_idx: continue
    if i >= max_idx: return data
    enc_src,enc_trg = map(sp_model.encode,[raw_src,raw_trg])
    if len(enc_src)<=max_len or len(enc_trg)<=max_len:
      data.append((enc_src, enc_trg))
    
class data_chunk_iter:
  '''
  this class enables its instances to load and encode parallel data in equally-sized chunks to satisfy 
  memory constraints.
  '''
  def __init__(self,src_file,trg_file,sp_model,max_len,chunk_size=None):
    self.src_file = src_file
    self.trg_file = trg_file
    self.max_len = max_len
    self.sp_model = sp_model
    self.skipped_chunks = 0
    self.size = count_examples(src_file,sp_model,max_len)
    unfiltered_size = count_examples(src_file)
    if chunk_size is not None:
        start_idxs = [idx for idx in range(0,unfiltered_size,chunk_size)]
        intervals = [(start,start+chunk_size) for start in start_idxs if start+chunk_size<=unfiltered_size]
        if intervals[-1][1] < unfiltered_size:
            intervals.append((intervals[-1][1],unfiltered_size))
    else:
       intervals = [(0,unfiltered_size)]
    self.intervals = intervals
  def __iter__(self):
    self.intervals_iter = iter(self.intervals[self.skipped_chunks:])
    return self
  def __next__(self): 
    return create_data_chunk(self.src_file,self.trg_file,self.sp_model,self.max_len,next(self.intervals_iter))
  def __len__(self):
     return self.size
  def skip_chunks(self,skipped_chunks):
     self.skipped_chunks = skipped_chunks

class TokenSizeBatchSampler:
    '''
    This class can be used as a batch sampler to create batches with equal numbers 
    of tokens as opposed to equal numbers of samples
    ''' 
    def __init__(self,dataset,tokens_size,sampler,samples_skipped=0):
        self.index_sampler = sampler
        self.tokens_size = tokens_size
        self.dataset = dataset
        self.samples_skipped = samples_skipped
    def __iter__(self):
       self.index_sampler_iter = iter(self.index_sampler)
       for i in range(self.samples_skipped):
          next(self.index_sampler)
       return self
    def __next__(self):
            indices = []
            total = 0
            while total >= self.tokens_size:
                try:
                    idx = next(self.index_sampler_iter)
                except StopIteration:
                    if indices == []:
                       raise StopIteration
                    return indices
                indices.append(idx)
                for seq in self.dataset[idx]:
                    total += len(seq)
            return indices
    def __len__(self):
       return len(self.dataset)                     

from torch.nn.utils.rnn import pad_sequence

def prepare_dataloader(dataset,batch_size,sampler,bos_id,eos_id,pad_id,samples_skipped=0):
    def construct_batch(data_batch):
        src_batch, trg_batch = [], []
        for (src_sent, trg_sent) in data_batch:
            src_batch.append(torch.cat([torch.tensor([bos_id]), torch.tensor(src_sent), torch.tensor([eos_id])], dim=0))
            trg_batch.append(torch.cat([torch.tensor([bos_id]), torch.tensor(trg_sent), torch.tensor([eos_id])], dim=0))
        src_batch = pad_sequence(src_batch, padding_value=pad_id)
        trg_batch = pad_sequence(trg_batch, padding_value=pad_id)
        return src_batch.transpose(0,1), trg_batch.transpose(0,1)
    batch_sampler = TokenSizeBatchSampler(dataset,batch_size,sampler,samples_skipped) 
    batch_iter = DataLoader(dataset,batch_sampler=batch_sampler, collate_fn=construct_batch)
    return batch_iter

    
