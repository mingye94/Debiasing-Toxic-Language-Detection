data_dir="./data/toxic"
glove_path="./embeddings/glove.6B.100d.txt"

python3 toxic_demo.py --data_dir $data_dir --glove_path $glove_path --num_envs 4 --log_path "logs/mention/toxic.env4" --env_type "mention"
