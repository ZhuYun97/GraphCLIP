import argparse

class Arguments:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        # self.parser.add_argument('--random_seeds', type=int, help='random seed', default=0)
        # self.parser.add_argument("--seed", type=int, default=0)
        # self.parser.add_argument('--config', type=str, help="the config file", default='./configs/cora/engine.yaml')
        # Dataset
        self.parser.add_argument('--dataset', type=str, help="dataset name", default='cora')
        
        # Model configuration
        self.parser.add_argument('--layer_num', type=int, help="the number of encoder's layers", default=2)
        self.parser.add_argument('--hidden_size', type=int, help="the hidden size", default=64)
        self.parser.add_argument('--dropout', type=float, help="dropout rate", default=0.5)
        self.parser.add_argument('--activation', type=str, help="activation function", default='relu', 
                                 choices=['relu', 'elu', 'hardtanh', 'leakyrelu', 'prelu', 'rrelu'])
        # self.parser.add_argument('--use_bn', action='store_true', help="use BN or not")
        self.parser.add_argument('--last_activation', action='store_true', help="the last layer will use activation function or not")
        self.parser.add_argument('--model', type=str, help="model name", default='GNN', 
                                 choices=['GNN'])
        self.parser.add_argument('--norm', type=str, help="the type of normalization, id denotes Identity(w/o norm), bn is batchnorm, ln is layernorm", default='id', 
                                 choices=['id', 'bn', 'ln'])
        self.parser.add_argument('--encoder', type=str, help="the type of encoder", default='GCN_Encoder', 
                                 choices=['GCN_Encoder', 'GAT_Encoder', 'SAGE_Encoder', 'GIN_Encoder', 'MLP_Encoder', 'GCNII_Encoder'])
        # Training settings
        self.parser.add_argument('--optimizer', type=str, help="the kind of optimizer", default='adam', 
                                 choices=['adam', 'sgd', 'adamw', 'nadam', 'radam'])
        self.parser.add_argument('--lr', type=float, help="learning rate", default=1e-3)
        self.parser.add_argument('--weight_decay', type=float, help="weight decay", default=5e-4)
        self.parser.add_argument('--epochs', type=int, help="training epochs", default=200)
        self.parser.add_argument('--batch_size', type=int, help="the batch size", default=256)
        # Early stopping
        self.parser.add_argument('--earlystop', action='store_true', help="earlystop")
        self.parser.add_argument('--patience', type=int, help="the patience of counting", default=20)
        self.parser.add_argument('--dynamic_p', type=int, help="the patience used in dynamic early exiting", default=20)
        
        # Processing node attributes
        self.parser.add_argument('--llm', action='store_true', help="use the output of llm as node features")
        self.parser.add_argument('--peft', type=str, help="the type of peft", default='lora', 
                                 choices=['lora', 'prefix', 'prompt', 'adapter', 'ia3'])
        self.parser.add_argument('--lm_type', type=str, help="the type of lm", default='sbert', 
                                 choices=['tiny', 'sbert', 'deberta', 'bert', 'e5', 'llama2', 'llama3', 'llama2-14', 'qwen2', 'qwen2.5-0.5b', 'tiny', 'sbert2'])
        
        # used for sampling
        self.parser.add_argument('--subsampling', action='store_true', help="subsampling, training with subgraphs")
        self.parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.5)
        self.parser.add_argument('--walk_steps', type=int, help="the steps of random walking", default=64)
        self.parser.add_argument('--k', type=int, help="the hop of neighboors", default=1)
        self.parser.add_argument('--sampler', type=str, help="the choice of sampler, random walk or k-hop sampling", default='rw', 
                                 choices=['rw', 'khop', 'shadow'])
        
        # dynamic early exit
        self.parser.add_argument('--early', action='store_true', help="the sign of dynamic early exit")
        # prompt type
        self.parser.add_argument('--prompt', type=str, help="the type of prompt tuning", default='gppt', 
                                 choices=['gppt', 'graphprompt', 'prog', 'gpf'])
        
    def parse_args(self):
        return self.parser.parse_args()