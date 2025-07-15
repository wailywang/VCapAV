 #!/bin/bash

# cd scratch place
# cd data/

###for small file
# FILEID='1uqH1Chlu-Rrkhad6vN2y5nhTk_7A3KMt'
# FILENAME='hmpd.tar.gz'
# wget --no-check-certificate "https://docs.google.com/uc?export=download&id=${FILEID}" -O ${FILENAME}


# # Download large file from Google Drive
# filename='straight.tar.gz'
# fileid='1oBRRsjEEBUheFCbZAI1VBVwDZVLp-rbm'
# wget --load-cookies ~/tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies ~/tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://drive.google.com/uc?export=download&id=${fileid}' -O- | sed -rn 's/.confirm=([0-9A-Za-z_]+)./\1\n/p')&id=${fileid}" -O ${filename} && rm -rf ~/tmp/cookies.txt


# # Unzip
# unzip -q ${filename}
# rm ${filename}
# cd
# https://drive.google.com/file/d/1uqH1Chlu-Rrkhad6vN2y5nhTk_7A3KMt/view?usp=share_link
# https://drive.google.com/file/d/1oBRRsjEEBUheFCbZAI1VBVwDZVLp-rbm/view?usp=share_link
