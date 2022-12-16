# embedding_ELECTRA
embedding each book using books title, keywords, isbn to 256 latent vector

example)
terminal: $> python embedding.py books.csv embedding.csv

books.csv must contain ['title','main_keywords','isbn'] columns
