import glob
import os
import pandas as pd

#HoverNet needs to be configured in advance: https://github.com/vqdang/hover_net
#Here, the path to the HoverNet configuration is set to: /software/hover_net
samples = glob.glob("/patches_result/*/*/*")#folder:patches_result/resistant（Classification）/2404992（Patient ID）/1（WSI ID）;
for sample in samples:
    with open('/software/hover_net/hovernet_sh.sh', 'a', encoding='utf-8') as f:
        f.write(f'python run_infer.py \\\n')
        f.write(f"--gpu='1' \\\n")
        f.write(f'--nr_types=6 \\\n')
        f.write(f'--type_info_path=type_info.json \\\n')
        f.write(f'--batch_size=32 \\\n')
        f.write(f'--model_mode=fast \\\n')
        f.write(f'--model_path=/software/hover_net/hovernet_fast_pannuke_type_tf2pytorch.tar \\\n')
        f.write(f'--nr_inference_workers=1 \\\n')
        f.write(f'--nr_post_proc_workers=2 \\\n')
        f.write(f'tile \\\n')
        f.write(f"--input_dir=/patches_result/{sample.split('/')[-3]}/{sample.split('/')[-2]}/{sample.split('/')[-1]}/ \\\n")
        f.write(f"--output_dir=/hovernet/hovernet-seg/{sample.split('/')[-3]}/{sample.split('/')[-2]}/{sample.split('/')[-1]}/ \\\n")
        f.write(f'--mem_usage=0.2 \\\n')
        f.write(f'--draw_dot \\\n')
        f.write(f'--save_qupath\n')
        f.write(f'\n')






