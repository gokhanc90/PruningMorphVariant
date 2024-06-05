
import torch

class Vector2:

    def __init__(self,args):
        self.vectors = []
        model_class = args['model_class']
        tokenizer_class = args['tokenizer_class']
        pretrained_weights = args['pretrained_weights']

        self.tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
        self.model = model_class.from_pretrained(pretrained_weights,output_hidden_states = True)
        self.model.eval()

    def bertVectorizer(self, words ):


        input_ids = []
        attention_masks = []
        segments_ids = []
        for w in range(len(words)):

          encoded_dict = self.tokenizer.encode_plus(
                        words[w],                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                        max_length = 16,           # Pad & truncate all sentences.
                        padding='longest',
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                        return_length=True,
                        truncation=True,
                   )

          input_ids.append(encoded_dict['input_ids'])

          attention_masks.append(encoded_dict['attention_mask'])

        max_len = 0
        for i in input_ids:
            if list(i.shape)[1] > max_len:
                max_len = list(i.shape)[1]
        padded=[]
        for i in input_ids:
          l=list(i.shape)[1]
          p=torch.nn.ZeroPad2d((0, max_len - l, 0, 0))
          padded.append(p(i))

        input_ids = padded

        padded=[]
        for i in attention_masks:
          l=list(i.shape)[1]
          p=torch.nn.ZeroPad2d((0, max_len - l, 0, 0))
          padded.append(p(i))

        attention_masks = padded

        # Convert the lists into tensors.
        tokens_tensor = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)



        with torch.no_grad():
            outputs = self.model(tokens_tensor,return_dict=True)
#            print(outputs)
            last_hidden_states=outputs['last_hidden_state']
            hidden_states = outputs['hidden_states']

        word_embeddings = torch.stack(hidden_states, dim=0)


        vectors = word_embeddings[-1][:, 0, :].numpy()

#
        self.vectors = vectors


from scipy import spatial
import pandas as pd

import transformers
import numpy as np


args ={
    'model_class' : transformers.BertModel,
    'tokenizer_class' : transformers.BertTokenizer,
    'pretrained_weights' : 'bert-base-uncased'
}

args2 ={
    'model_class' : transformers.DistilBertModel,
    'tokenizer_class' : transformers.DistilBertTokenizer,
    'pretrained_weights' : 'distilbert-base-uncased'
}

vectorizer = Vector2(args)


def getSim(word1,word2):

  vectorizer.bertVectorizer([word1,word2])
  vectors_bert = vectorizer.vectors
  return 1-spatial.distance.cosine(vectors_bert[0], vectors_bert[1])


df = pd.read_excel("npmi-CW09B.xlsx",sheet_name='SnowballEng',header=0)
print(f'Number of Pairs {df.shape[0]}\n')
df.sample(10)
df['bert-base-uncased'] = np.nan



for index, row in df.iterrows():
   # print(index,row['Term'], row['Morph'])
   sim = getSim(row['Term'], row['Morph'])
   df.at[index , 'bert-base-uncased'] =  sim
#   print(index)

df.sample(10)

with pd.ExcelWriter('npmi-CW09B-Bert.xlsx', mode='w') as writer:
    df.to_excel(writer, sheet_name='SnowballEng', index=False)
