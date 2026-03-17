from typing import List

def recursive_token_chunking(text: str, max_tokens: int) -> List[str]:
    """
    Splits the input text into chunks based on a maximum token limit.
    
    Args:
        text (str): The input text to be chunked.
        max_tokens (int): The maximum number of tokens per chunk.
        
    Returns:
        List[str]: A list of text chunks.
    """
    import nltk
    from nltk.tokenize import word_tokenize

    # Ensure that NLTK resources are available
    nltk.download('punkt', quiet=True)

    tokens = word_tokenize(text)
    chunks = []
    current_chunk = []

    for token in tokens:
        current_chunk.append(token)
        if len(current_chunk) >= max_tokens:
            chunks.append(' '.join(current_chunk))
            current_chunk = []

    # Add any remaining tokens as the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks