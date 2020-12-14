from typing import Dict, List, Optional

from tqdm import tqdm


def get_tokenized_texts(tokenizer: , texts: List[str], tokenizer_encode_kwargs: Optional[Dict[str, any]] = None):
    if tokenizer_encode_kwargs is None:
        tokenizer_encode_kwargs = {
            "max_length": 50,
            "add_space_before_punct_symbol": True,
            "pad_to_max_length": True,
            "add_special_tokens": True,
            "return_tensors": "pt"
        }
    tokenized_texts = []
    for text in tqdm(texts):
        encoded_prompt = tokenizer.encode(text, **tokenizer_encode_kwargs)
        tokenized_texts.append(encoded_prompt)
    return tokenized_texts
