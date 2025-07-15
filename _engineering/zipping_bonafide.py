import os
import zipfile
import tqdm
# 定义文件路径
wav_scp_path = '/Work29/wwm1995/SMIIP/Anti_Spoof/ASVspoof5/data/train/wav.scp'
utt2label_path = '/Work29/wwm1995/SMIIP/Anti_Spoof/ASVspoof5/data/train/utt2label'
output_zip_path = '/Corpus3/yikang/ASVspoof5/train_bonafide_flacs.zip'

# 读取utt2label文件并获取标签为bonafide的utterance
bonafide_utterances = []
with open(utt2label_path, 'r') as f:
    for line in f:
        utt, label = line.strip().split()
        if label == 'bonafide':
            bonafide_utterances.append(utt)

# 读取wav.scp文件并获取对应的文件路径
flac_files = []
with open(wav_scp_path, 'r') as f:
    for line in f:
        utt, flac_path = line.strip().split()
        if utt in bonafide_utterances:
            flac_files.append(flac_path)

# 创建一个zip文件并将对应的flac文件添加进去
with zipfile.ZipFile(output_zip_path, 'w') as zipf:
    for flac_file in tqdm.tqdm(flac_files):
        zipf.write(flac_file, os.path.basename(flac_file))

print(f'Bonafide FLAC files have been zipped into {output_zip_path}')
