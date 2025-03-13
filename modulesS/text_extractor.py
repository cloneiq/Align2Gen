import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

class TextExtractor(nn.Module):
  def __init__(self, args):
    super(TextExtractor, self).__init__()
    self.embed_size = args.embed_size
    self.no_txtnorm = args.no_txtnorm

    # Load pretrained BERT model and tokenizer
    bert_model_name = 'bert-base-uncased'
    self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    self.bert = BertModel.from_pretrained(bert_model_name)

    # Project BERT hidden size to the desired embedding size
    self.fc = nn.Linear(self.bert.config.hidden_size, self.embed_size)

  def forward(self, text):
    """
        text: List of strings (captions)
    """
    # Tokenize the input captions and convert to input IDs
    inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)

    # Move input tensors to the appropriate device (GPU if available)
    device = next(self.parameters()).device  # Get device from the model parameters
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)

    # Forward pass through BERT model
    outputs = self.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

    # Extract all token embeddings
    cap_emb = outputs.last_hidden_state  # Shape: [batch_size, max_length, hidden_size]

    # Project to the same embedding size as the image features
    cap_emb = self.fc(cap_emb)  # Now shape is [batch_size, max_length, embed_size]

    # Get the actual lengths of the input captions
    cap_len = inputs['input_ids'].ne(self.tokenizer.pad_token_id).sum(dim=1)  # Count non-padding tokens

    # Optional normalization in the joint embedding space
    if not self.no_txtnorm:
      cap_emb = F.normalize(cap_emb, p=2, dim=-1)

    return cap_emb, cap_len  # Return embedding and lengths

#---------------------texts_list------------------------

class TextListExtractor(nn.Module):
    def __init__(self, args):
        super(TextListExtractor, self).__init__()
        self.embed_size = args.embed_size
        self.no_txtnorm = args.no_txtnorm  # bool

        # Load pretrained BERT model and tokenizer
        bert_model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Project BERT hidden size to the desired embedding size
        self.fc = nn.Linear(self.bert.config.hidden_size, self.embed_size)

    def forward(self, text_batch):
        """
        text_batch: List of lists of strings (captions grouped for each image)
        """
        cap_emb_list = []
        cap_len_list = []

        for text_group in text_batch:
            inputs = self.tokenizer(text_group, return_tensors='pt', padding=True, truncation=True, max_length=512)

            # Move input tensors to the appropriate device (GPU if available)
            device = next(self.parameters()).device  # Get device from the model parameters
            inputs['input_ids'] = inputs['input_ids'].to(device)
            inputs['attention_mask'] = inputs['attention_mask'].to(device)

            # Forward pass through BERT model
            outputs = self.bert(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

            # Extract all token embeddings
            cap_emb = outputs.last_hidden_state  # Shape: [batch_size, max_length, hidden_size]

            # Project to the same embedding size as the image features
            cap_emb = self.fc(cap_emb)  # Now shape is [batch_size, max_length, embed_size]

            # Get the actual lengths of the input captions
            cap_len = inputs['input_ids'].ne(self.tokenizer.pad_token_id).sum(dim=1)  # Count non-padding tokens

            # Optional normalization in the joint embedding space
            if not self.no_txtnorm:
                cap_emb = F.normalize(cap_emb, p=2, dim=-1)

            cap_emb_list.append(cap_emb)
            cap_len_list.append(cap_len)


        return cap_emb_list, cap_len_list