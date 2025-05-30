
from rag_tokenizer import tokenize, fine_grained_tokenize
import re

filename = "python,membership inference.pdf"


print(tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename)))

print(fine_grained_tokenize(tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename))))
