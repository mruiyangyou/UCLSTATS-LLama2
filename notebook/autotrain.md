# autotrain the llama model

1. Use Meta's with normal shard
```bash
autotrain llm --train --project_name my-llm --model meta-llama/Llama-2-7b-hf --data_path . --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 5 --num_train_epochs 3 --trainer sft --push_to_hub --repo_id mRuiyang/uclstats-llama
```

2. Use llama2 model with less shard
```bash
!autotrain llm --train --project_name my-llama3 --model abhishek/llama-2-7b-hf-small-shards --data_path . --use_peft --use_int4 --learning_rate 2e-4 --train_batch_size 5 --num_train_epochs 3 --trainer sft --push_to_hub --repo_id mRuiyang/uclstats-llama > training.log &
```
