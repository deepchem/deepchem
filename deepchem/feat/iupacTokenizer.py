# Requirements
from transformers import BertTokenizer
import re
import os
import collections

def load_vocab_file(vocab_file):
# Loads a vocabulary file into a dictionary
  vocabulary=collections.OrderedDict()
  with open(vocab_file,'r',encoding='utf-8') as file:
    tokens=file.readlines()
  for idx,tok in enumerate(tokens):
    tok=tok.rstrip('\n')
    vocabulary[tok]=idx
  return vocabulary

class iupacTokenizer(BertTokenizer):
  def __init__(self,
              vocab_file:str='',
              unk_token="[UNK]",
              sep_token="[SEP]",
              cls_token="[CLS]",
              mask_token="[MASK]",
              pad_token="[PAD]",
              **kwargs):
    super().__init__(vocab_file,
                    unk_token=unk_token,
                    sep_token=sep_token,
                    cls_token=cls_token,
                    mask_token=mask_token,
                    pad_token=pad_token,
                    **kwargs)
    # Loading vocabulary file
    if not os.path.isfile(vocab_file):
      raise ValueError(f"File not found at path {vocab_file}")
    self.vocab=load_vocab_file(vocab_file)
    self.ids_to_tokens=collections.OrderedDict([
      (ids,tok) for tok,ids in self.vocab.items()
    ])
    # Common prefixes used in iupac naming
    self.iupac_prefixes=['trans-','cis-','N-','O-','hydroxy-','alkoxy-','oxo-','carboxy-','alkoxycarbonyl-','amino-','cyano-','carbamoyl-']
    # Common suffixes used in iupac naming
    self.iupac_suffixes=['-ol','-one','-oic acid','-ether','-al','-oate','-amine','-nitrile','-amide','-yne','-ane','-ene']
    
  def _tokenize(self,seq):
    tokens=[]
    # Initial split by common delimiters
    split_tokens=re.split(r"([-,\(])",seq)
    split_tokens=[i.strip() for i in split_tokens if i.strip()]
    
    for tok in split_tokens:
    # Handling locants
      if re.match(r"^\d+-",tok):
        match=re.match(r"^(\d+-)(.*)",tok)
        tokens.append(match.group(1))
        tok=match.group(2)
        
      prefix=False
      for prefix in self.iupac_prefixes:
        # Handling prefixes
        if tok.startswith(prefix):
          tokens.append(prefix)
          tok=tok[len(prefix):]
          prefix=True
          break
        
      suffix=False
      for suffix in self.iupac_suffixes:
      # Handling suffixes
        if tok.endswith(suffix):
          base=tok[:-len(suffix)]
          if base:
            tokens.append(base)
          tokens.append(suffix)
          suffix=True
          break
        
      if not prefix and not suffix and tok:
      # For rest parts, apply WordPiece
        word_toks=super()._tokenize(tok)
        tokens.extend(word_toks)
        
    return tokens
  
  def _convert_token_to_id(self,tok:str):
  # Converts a token(str/unicode) into an id using vocab
    return self.vocab.get(tok,self.vocab.get(self.unk_token))
  
  def _convert_id_to_token(self, index:int):
  # Converts an index(integer) into a token(str/unicode) using vocab
    return self.ids_to_tokens.get(index,self.unk_token)
  
  def convert_tokens_to_string(self, tokens:list):
  # Converts a sequence of tokens(string) to a single string
    return " ".join(tokens).replace('##'," ").strip()
  
  def add_special_tokens(self,tokens:list):
  # Adds special tokens to the sequence for sequence classification tasks. A BERT sequence has the following format : [CLS]X[SEP]
    return [self.cls_token]+tokens+[self.sep_token]
  
  def add_special_tokens_ids(self,token_ids:list):
  # Adds special token ids to the sequence for sequence classification tasks. A BERT sequence has the following format : [CLS]X[SEP]
    return [self.cls_token_id]+token_ids+[self.sep_token_id]
  
  def add_padding_tokens(self,token_ids:list,length:int):
  # Adds padding tokens to return a sequence of length max_length
    padding=[self.pad_token_id]*(length-len(token_ids))
    return token_ids+padding    