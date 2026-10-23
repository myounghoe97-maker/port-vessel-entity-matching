"""Public extraction: frozen normalize_name only; research runner omitted."""
import re
import unicodedata
import pandas as pd

def normalize_name(value):
    if value is None or pd.isna(value):
        return ''
    text=unicodedata.normalize('NFKC',str(value)).upper().strip()
    # Remove exactly all Unicode whitespace and the four specified separators.
    # All other characters remain; no translation, numeral conversion or lexicon.
    return re.sub(r'[\s\-_.\/]+','',text)
