import math
import re
import Levenshtein
from typing import Tuple
from collections import Counter

def separate_digits(token: str) -> Tuple[str, str]:
    nums = re.findall(r"\d+", token)
    letters = re.sub(r"\d+", "", token)
    return letters, "".join(nums)


def qgram_jaccard(s1: str, s2: str, q: int = 3) -> float:
    if not s1 or not s2:
        return 0.0

    s1 = f"#{s1.lower()}#"
    s2 = f"#{s2.lower()}#"

    q1 = Counter(s1[i:i + q] for i in range(len(s1) - q + 1))
    q2 = Counter(s2[i:i + q] for i in range(len(s2) - q + 1))

    n1 = sum(q1.values())
    n2 = sum(q2.values())

    intersection = sum((q1 & q2).values())
    qgram_distance = n1 + n2 - 2 * intersection

    return (n1 + n2 - qgram_distance) / (n1 + n2) if (n1 + n2) else 0.0


def cosine_words(a: str, b: str) -> float:
    if not a or not b:
        return 0.0

    set_a = set(a.split())
    set_b = set(b.split())

    dot = len(set_a & set_b)
    sa = len(set_a)
    sb = len(set_b)

    return dot / (math.sqrt(sa) * math.sqrt(sb)) if sa and sb else 0.0


def normalized_lev(a: str, b: str) -> float:
    if not a or not b:
        return 1.0

    dist = Levenshtein.distance(a, b)
    m = max(len(a), len(b))

    return dist / m if m else 0.0


def avg_levenshtein_sim(words1, words2, mw_flag: bool) -> float:

    if not words1 or not words2:
        return 0.0

    numerator = 0.0
    denominator = 0.0

    for w1 in words1:
        non1, num1 = separate_digits(w1)

        for w2 in words2:
            non2, num2 = separate_digits(w2)

            if (not mw_flag) or (
                mw_flag and
                (1 - normalized_lev(non1, non2)) > 0.5 and
                num1 == num2
            ):
                sim = 1 - normalized_lev(w1, w2)
                weight = len(w1) + len(w2)

                numerator += sim * weight
                denominator += weight

    return numerator / denominator if denominator > 0 else 0.0

