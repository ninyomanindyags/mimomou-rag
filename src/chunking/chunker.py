"""Text splitting / chunking dokumen."""
from langchain_text_splitters import RecursiveCharacterTextSplitter # pake text splitter dari langchain untuk memecah dokumen menjadi potongan-potongan kecil

from src.utils.helpers import load_config 

def chunk_documents(
        documents, 
        chunk_size: int | None = None, 
        chunk_overlap: int | None = None
    ):
        config = load_config() # buat baca nilai di file config.yaml
        chunk_size = chunk_size or config["chunking"]["chunk_size"]
        chunk_overlap = chunk_overlap or config["chunking"]["chunk_overlap"]

        # buat inisialisasi text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, # maks jumlah karakter setiap chunk
            chunk_overlap=chunk_overlap, # jumlah karakter diulang di awal chunk berikutnya
        )
        return splitter.split_documents(documents) # memecah dokumen menjadi potongan-potongan kecil (chunk)