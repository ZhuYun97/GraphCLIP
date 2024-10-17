import torch


def calculate_loss(graph_logits, text_logits, criterion):
    batch_size = graph_logits.shape[0]
    gt = torch.arange(batch_size).to(graph_logits.device)
    total_train_image_loss = criterion(graph_logits, gt)
    total_train_text_loss = criterion(text_logits, gt)
    
    total_train_loss = (total_train_image_loss + total_train_text_loss)/2
    return total_train_loss

def create_logits(x1,x2,logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True)
    x2 = x2 / x2.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_x1 =  logit_scale*x1 @ x2.t()
    logits_per_x2 =  logit_scale*x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2


class TextCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(TextCLIP, self).__init__()
        self.model = model
        
    def forward(self,input_ids, token_type_ids, attention_mask):
        # return self.model.encode_text(input_ids, token_type_ids, attention_mask)
        return self.model.encode_text(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    
    
class GCLIP(torch.nn.Module):
    def __init__(self, model) :
        super(GCLIP, self).__init__()
        self.model = model
        
    def forward(self,batch):
        return self.model.encode_graph(batch)