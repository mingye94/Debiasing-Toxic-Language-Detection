data_dir="./data/Toxic"
glove_path="./embeddings/glove.6B.100d.txt"

python3 toxic_demo.py --data_dir $data_dir --glove_path $glove_path --num_envs 4 --log_path "logs/toxic.mention1"
