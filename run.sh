# Reference upstream repository: https://github.com/longinyu/ksivi
# This file is part of the KPG overlay package.

### toy example
python sivipathstein-is_2d.py --config "multimodal.yml" 
python sivipathstein-is_2d.py --config "x_shaped.yml" 
python sivipathstein-is_2d.py --config "banana.yml" 

python sivipathstein-is_lr.py --config LRwaveform.yml --baseline_sample SGLD_trace/parallel_SGLD_LRwaveform.pt 

python sivipathstein-is_langevin_post.py --config kernel_sivi_langevin_post.yml 
