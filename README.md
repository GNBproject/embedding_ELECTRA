# embedding_ELECTRA
embedding each book using books title, keywords, isbn to 256 latent vector

example)
terminal: 
on embedding_ELECTRA directory..

% ipython3 src/embedding.py data/final_processing.csv embedding.csv

if ipython is not intalled:
% sudo apt install ipython3

books.csv must contain ['title','main_keywords','isbn'] columns
