def create(start, end):
    to_write = ''
    for i in range(1, 1+end-start+1):
        origin='''[%d]
label = %d
model_type = spert
model_path = data/save/BioTrain.json/2021-09-14_21:42:25.902585/final_model
tokenizer_path = data/save/BioTrain.json/2021-09-14_21:42:25.902585/final_model
dataset_path = data/large_data/%d/tests.json
types_path = data/datasets/bioPre/biotypes.json
eval_batch_size = 1
rel_filter_threshold = 0.4
size_embedding = 25
prop_drop = 0.1
max_span_size = 10
store_predictions = true
store_examples = false
sampling_processes = 1
max_pairs = 1000
log_path = data/final_res/%d'''%(i,start+i-1,start+i-1,start+i-1,)
        to_write+=origin
        to_write+='\n'
        to_write+='\n'
    with open('%dto%d.conf'%(start,end),'w') as f:
        f.write(to_write)

# 开始的文件，结束的文件
create(23,24)
