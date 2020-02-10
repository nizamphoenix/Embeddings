'''
credits: https://github.com/abhishekkrthakur
The following youtube link takes one through from setting up the machines to fine tune a BERT model
Source: https://www.youtube.com/watch?v=B_P0ZIXspOU
'''

import torch
import transformers
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn import model_selection
from transformers import AdamW,get_linear_schedule_with_warmup
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
import torch_xla.distributed.parallel_loader as pl
from scipy import stats
import sys
import warnings
warnings.filterwarnings('ignore')
class BERTBaseUncased(nn.Module):
    def __init__(self,bert_path):
        super(BERTBaseUncased,self).__init__()
        self.bert_path=bert_path
        self.bert=transformers.BertModel.from_pretrained(self.bert_path)
        self.bert_drop=nn.Dropout(0.3)
        self.out=nn.Linear(768,30)
    def forward(self,ids,mask,token_type_ids):
        _,o2=self.bert(ids,attention_mask=mask,token_type_ids=token_type_ids)
        bo=self.bert_drop(o2)
        return self.out(bo)
    
class BERTDatasetTraining:
    def __init__(self,qtitle,qbody,answer,targets,tokenizer,max_len):
        self.qtitle=qtitle
        self.qbody=qbody
        self.answer=answer
        self.tokenizer=tokenizer
        self.max_len=max_len
        self.targets=targets
    def __len__(self):
        return len(self.answer)
    def __getitem__(self,item):
        question_title=str(self.qtitle[item])
        question_body=str(self.qbody[item])
        answer=str(self.answer[item])
        inputs=self.tokenizer.encode_plus(
            question_title+" "+question_body,
            answer,
            add_special_tokens=True,
            max_length=self.max_len
        )    
        ids=inputs["input_ids"]
        token_type_ids=inputs["token_type_ids"]
        mask=inputs["attention_mask"]
        padding_len=self.max_len-len(ids)
        ids=ids+([0]*padding_len)
        token_type_ids=token_type_ids+([0]*padding_len)
        mask=mask+([0]*padding_len)
        return{
            "ids":torch.tensor(ids,dtype=torch.long),
                "mask":torch.tensor(mask,dtype=torch.long),
              "token_type_ids":torch.tensor(token_type_ids,dtype=torch.long),
               "targets":torch.tensor(self.targets[item,:],dtype=torch.float)
        }
    
def loss_fn(outputs,targets):
    return nn.BCEWithLogitsLoss()(outputs,targets)
   
def train_loop_fn(data_loader,model,optimizer,device,scheduler=None):
    model.train()
    for bi,d in enumerate(data_loader):
        ids=d["ids"]
        mask=d["mask"]
        token_type_ids=d["token_type_ids"]
        targets=d["targets"]
        ids=ids.to(device,dtype=torch.long)
        mask=mask.to(device,dtype=torch.long)
        token_type_ids=token_type_ids.to(device,dtype=torch.long)
        targets=targets.to(device,dtype=torch.float)
        optimizer.zero_grad()
        outputs=model(ids=ids,mask=mask,token_type_ids=token_type_ids)
        loss=loss_fn(outputs,targets)
        loss.backward()
        xm.optimizer_step(optimizer)
        if scheduler is not None:
            scheduler.step()
        if bi%10==0:
            xm.master_print(f"bi={bi},loss={loss}")
            
            
def eval_loop_fn(data_loader,model,device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    for bi,d in enumerate(data_loader):
        ids=d["ids"]
        mask=d["mask"]
        token_type_ids=d["token_type_ids"]
        targets=d["targets"]
        ids=ids.to(device,dtype=torch.long)
        mask=mask.to(device,dtype=torch.long)
        token_type_ids=token_type_ids.to(device,dtype=torch.long)
        targets=targets.to(device,dtype=torch.float)
        outputs=model(ids=ids,mask=mask,token_type_ids=token_type_ids)
        loss=loss_fn(outputs,targets)
        fin_targets.append(targets.cpu().detach().numpy())
        fin_outputs.append(outputs.cpu().detach().numpy())
        return np.vstack(fin_outputs),np.vstack(fin_targets)
    
def run(index):
    MAX_LEN=512
    TRAIN_BATCH_SIZE=16
    EPOCHS=50
    dfx=pd.read_csv("/home/nizamphoenix/dataset/train.csv").fillna("none")
    df_train,df_valid=model_selection.train_test_split(dfx,random_state=42,test_size=0.3)
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    sample=pd.read_csv("/home/nizamphoenix/dataset/sample_submission.csv")
    target_cols=list(sample.drop("qa_id",axis=1).columns)
    train_targets = df_train[target_cols].values
    valid_targets = df_valid[target_cols].values

    tokenizer=transformers.BertTokenizer.from_pretrained("/home/nizamphoenix/bert-base-uncased/")
    
    train_dataset=BERTDatasetTraining(
    qtitle=df_train.question_title.values,
    qbody=df_train.question_body.values,
    answer=df_train.answer.values,
    targets=train_targets,
    tokenizer=tokenizer,
    max_len=MAX_LEN
    )
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True
    )
    train_data_loader=torch.utils.data.DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        sampler=train_sampler
    )
    valid_dataset=BERTDatasetTraining(
    qtitle=df_valid.question_title.values,
    qbody=df_valid.question_body.values,
    answer=df_valid.answer.values,
    targets=valid_targets,
    tokenizer=tokenizer,
    max_len=MAX_LEN
    )
    valid_sampler = torch.utils.data.DistributedSampler(
        valid_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
    )
    valid_data_loader=torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=8,###change
        sampler=valid_sampler
    )
    
    device=xm.xla_device()
    
    lr = 2e-5 * xm.xrt_world_size()###change
    num_train_steps=int(len(train_dataset) / TRAIN_BATCH_SIZE / xm.xrt_world_size() * EPOCHS)
    model=BERTBaseUncased("/home/nizamphoenix/bert-base-uncased/").to(device)
    optimizer=AdamW(model.parameters(),lr=lr,eps = 1e-8)#eps = 1e-8: to prevent any division by zero 
    scheduler=get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )

    for epoch in range(EPOCHS):
        para_loader=pl.ParallelLoader(train_data_loader,[device])
        train_loop_fn(para_loader.per_device_loader(device),model,optimizer,device,scheduler)
        
        para_loader=pl.ParallelLoader(valid_data_loader,[device])
        o,t=eval_loop_fn(para_loader.per_device_loader(device),model,device)

        spear=[]
        for jj in range(t.shape[1]):
            p1=list(t[:,jj])
            p2=list(o[:,jj])
            coef,_=np.nan_to_num(stats.spearmanr(p1,p2))
            spear.append(coef)
        spear=np.mean(spear)
        xm.master_print(f"epoch={epoch},spearman={spear}")
        xm.save(model.state_dict(),"model3.bin")
        

if __name__=="__main__":
    #sys.path.insert(0, "/home/nizamphoenix/transformers/transformers-master/transformers/")
    xmp.spawn(run,nprocs=8)
