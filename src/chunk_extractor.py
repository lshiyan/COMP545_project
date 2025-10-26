from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from typing import List

class ChunkExtractor:
    def __init__(self):
        """
        Initilizes ChunkExtractor object. Uses hugging face sentence embedding model.
        """
        self.embeddings = HuggingFaceEmbeddings(model="sentence-transformers/all-MiniLM-l6-v2")
        self.text_splitter = SemanticChunker(embeddings=self.embeddings)

    def extract_chunks(self, text:str) -> List[str]:
        """
        Splits text into semantic chunks.

        Args:
            text (str): The text to split.

        Returns:
            List[str]: List of text chunks.
        """
        chunks = self.text_splitter.split_text(text)
        return chunks