import argparse
from load_data import DataLoader
import torch
from base_model import BaseModel

#I've changed a few things in other scripts to make this work on CPU by changing parts that say .cuda() to .to('cpu')
parser = argparse.ArgumentParser(description="Parser for EmerGNN")
parser.add_argument('--task_dir', type=str, default='./', help='the directory to dataset')
parser.add_argument('--dataset', type=str, default='S0', help='the directory to dataset')
parser.add_argument('--lamb', type=float, default=7e-4, help='set weight decay value')
parser.add_argument('--gpu', type=int, default=-1, help='GPU id to load.')
parser.add_argument('--n_dim', type=int, default=128, help='set embedding dimension')
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--load_model', default=True, action='store_true') #make this True
parser.add_argument('--lr', type=float, default=0.03, help='set learning rate')
parser.add_argument('--n_epoch', type=int, default=100, help='number of training epochs')
parser.add_argument('--n_batch', type=int, default=512, help='batch size')
parser.add_argument('--epoch_per_test', type=int, default=5, help='frequency of testing')
parser.add_argument('--test_batch_size', type=int, default=16, help='test batch size')
parser.add_argument('--seed', type=int, default=1234)

args = parser.parse_args()
torch.cuda.set_device(args.gpu)
device = torch.device('cpu')
dataloader = DataLoader(args)
eval_ent, eval_rel = dataloader.eval_ent, dataloader.eval_rel
KG = dataloader.KG
print(KG)
args.all_ent, args.all_rel, args.eval_rel = dataloader.all_ent, dataloader.all_rel, dataloader.eval_rel
#for S0 only:
args.lr = 0.01
args.lamb = 0.000001
args.n_dim = 32
args.n_batch = 32
args.length = 3
args.feat = 'E'

model = BaseModel(eval_ent, eval_rel, args, entity_vocab=dataloader.id2entity, relation_vocab=dataloader.id2relation)

triplet_to_test = torch.tensor([25, 26]) #Change this to map drugs to their IDs

pred = model.test_single(triplet_to_test, KG)
print('Prediction on '+str(triplet_to_test[0]) + ' and '+str(triplet_to_test[1]))
print(pred)