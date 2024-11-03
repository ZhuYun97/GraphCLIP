# generate subgraphs for target datasets: subgraph---label sentence pair
python gen_target_subg.py --dataset cora --sampler khop --k 3
# python gen_target_subg.py --dataset citeseer --sampler khop --k 3
python gen_target_subg.py --dataset wikics --sampler rw --walk_step 64
python gen_target_subg.py --dataset instagram --sampler rw --walk_step 256
python gen_target_subg.py --dataset photo --sampler rw --walk_step 256
python gen_target_subg.py --dataset computer --sampler rw --walk_step 256
python gen_target_subg.py --dataset history --sampler rw --walk_step 32
