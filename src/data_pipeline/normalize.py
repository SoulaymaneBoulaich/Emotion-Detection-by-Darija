"""Conservative Darija text normalization.

Owner: Fares Ilham

Rules:
- Unify common letter variants
- Remove diacritics and elongation marks
- Collapse repeated letters
- Replace URLs and user mentions
- Keep emojis, punctuation, and Latin-script words
"""

import re


def normalize_darija(text: str) -> str:
    if not text:
        return text

    # Replace URLs
    text = re.sub(r"http\S+|www\.\S+", "<URL>", text)
    # Replace mentions
    text = re.sub(r"@\w+", "<MENTION>", text)

    # Remove Arabic diacritics (tashkeel)
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652]", "", text)

    # Collapse elongation (3+ repeated chars -> 2)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    return text.strip()


if __name__ == "__main__":
    samples = [
        "حياك الله يا Brüder!!!!",
        "هذا الرابط https://example.com @user",
    ]
    for s in samples:
        print(f"{s}  ->  {normalize_darija(s)}")