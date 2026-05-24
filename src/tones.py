"""
Vietnamese tone classification + rhyme extraction.
Used by preprocess.py, sample.py, and server.py for Phase 9 (rhyme conditioning).
"""

# ═══════════════════════════════════════════════════════════════
# TONE CLASSIFICATION
# ═══════════════════════════════════════════════════════════════

# Bằng (level): ngang, huyền
BANG = set(
    "aăâeêioôơuưy"                              # ngang (no diacritic)
    "àằầèềìòồờùừỳ"                              # huyền ( ` )
    "AĂÂEÊIOÔƠUƯY"
    "ÀẰẦÈỀÌÒỒỜÙỪỲ"
)

# Trắc (sharp): sắc, nặng, hỏi, ngã
TRAC = set(
    "áắấéếíóốớúứý"                              # sắc ( ´ )
    "ạặậẹệịọộợụựỵ"                              # nặng ( . )
    "ảẳẩẻểỉỏổởủửỷ"                              # hỏi ( ̉ )
    "ãẵẫẽễĩõỗỡũữỹ"                              # ngã ( ~ )
    "ÁẮẤÉẾÍÓỐỚÚỨÝ"
    "ẠẶẬẸỆỊỌỘỢỤỰỴ"
    "ẢẲẨẺỂỈỎỔỞỦỬỶ"
    "ÃẴẪẼỄĨÕỖỠŨỮỸ"
)


def get_tone(syllable: str) -> str:
    """Return 'B' (bằng) or 'T' (trắc). Scans TRAC first — toned vowels take priority."""
    for ch in syllable:
        if ch in TRAC:
            return "T"
    for ch in syllable:
        if ch in BANG:
            return "B"
    return "B"  # no marked vowel = ngang = bằng


def get_tone_sequence(line: str) -> str:
    """Return tone string, e.g. 'BBTBBT' for a 6-syllable line."""
    return "".join(get_tone(s) for s in line.strip().split())


# ═══════════════════════════════════════════════════════════════
# RHYME EXTRACTION
# ═══════════════════════════════════════════════════════════════

# Tone mark → base vowel (for stripping diacritics before rhyme extraction)
_TONE_MARKS = {
    "à": "a", "á": "a", "ả": "a", "ã": "a", "ạ": "a",
    "ằ": "ă", "ắ": "ă", "ẳ": "ă", "ẵ": "ă", "ặ": "ă",
    "ầ": "â", "ấ": "â", "ẩ": "â", "ẫ": "â", "ậ": "â",
    "è": "e", "é": "e", "ẻ": "e", "ẽ": "e", "ẹ": "e",
    "ề": "ê", "ế": "ê", "ể": "ê", "ễ": "ê", "ệ": "ê",
    "ì": "i", "í": "i", "ỉ": "i", "ĩ": "i", "ị": "i",
    "ò": "o", "ó": "o", "ỏ": "o", "õ": "o", "ọ": "o",
    "ồ": "ô", "ố": "ô", "ổ": "ô", "ỗ": "ô", "ộ": "ô",
    "ờ": "ơ", "ớ": "ơ", "ở": "ơ", "ỡ": "ơ", "ợ": "ơ",
    "ù": "u", "ú": "u", "ủ": "u", "ũ": "u", "ụ": "u",
    "ừ": "ư", "ứ": "ư", "ử": "ư", "ữ": "ư", "ự": "ư",
    "ỳ": "y", "ý": "y", "ỷ": "y", "ỹ": "y", "ỵ": "y",
}
_TONE_MARKS.update({k.upper(): v.upper() for k, v in list(_TONE_MARKS.items())})

_VOWELS = set("aăâeêioôơuưyAĂÂEÊIOÔƠUƯY")


def strip_tone(syllable: str) -> str:
    """Remove tone diacritics: 'buồn' → 'buôn', 'tình' → 'tinh'."""
    return "".join(_TONE_MARKS.get(c, c) for c in syllable)


def get_rhyme_group(syllable: str) -> str:
    """
    Extract rhyme group (vần) from a syllable.
    Everything from the last vowel onward, after stripping tone marks.

    'sen' → 'en', 'buồn' → 'uôn', 'thương' → 'ương'
    """
    base = strip_tone(syllable).lower()

    # Find last vowel position
    last_vowel = -1
    for i, c in enumerate(base):
        if c in _VOWELS:
            last_vowel = i

    if last_vowel < 0:
        return base

    return base[last_vowel:]


def get_luc_bat_tags(prompt: str) -> tuple:
    """
    Extract [RHYME:X] and [TONE:XXXXXX] tags for a 6-syllable Lục Bát prompt.
    Returns (rhyme_tag, tone_tag) — empty strings if prompt too short.
    """
    syls = prompt.strip().split()
    rhyme_tag = ""
    tone_tag = ""

    if len(syls) >= 6:
        rhyme = get_rhyme_group(syls[5])
        rhyme_tag = f"[RHYME:{rhyme}]"

    if len(syls) >= 2:
        seq = get_tone_sequence(prompt)
        tone_tag = f"[TONE:{seq[:6]}]"

    return rhyme_tag, tone_tag


def get_that_ngon_tags(prompt: str) -> tuple:
    """
    Extract [LINK2:X] and [DOIAM:XXXXXXX] tags for a 7-syllable Thất Ngôn prompt.
    LINK2: tone (B/T) of 2nd syllable.
    DOIAM: expected response tone pattern (opposite of prompt).
    Returns (link2_tag, doi_am_tag).
    """
    syls = prompt.strip().split()
    link2_tag = ""
    doi_am_tag = ""
    
    if len(syls) >= 2:
        link2_tag = f"[LINK2:{get_tone(syls[1])}]"
    
    if len(syls) >= 7:
        p_tones = get_tone_sequence(prompt)[:7]
        # Đối âm: response tones = opposite of prompt tones
        r_tones = ''.join('T' if t == 'B' else 'B' for t in p_tones)
        doi_am_tag = f"[DOIAM:{r_tones}]"
    
    return link2_tag, doi_am_tag


def get_doi_tho_tags(six_line: str, eight_line: str) -> tuple:
    """
    Extract [RHYME:X] and [TONE:XXXXXX] tags for đối thơ.
    
    [RHYME:X] — from position 8 of the 8-syllable line (chain rhyme)
    [TONE:XXXXXX] — tone pattern of the 6-syllable line
    
    Args:
        six_line: the 6-syllable line of the last input couplet
        eight_line: the 8-syllable line of the last input couplet
    
    Returns (rhyme_tag, tone_tag).
    """
    import re
    rhyme_tag = ""
    tone_tag = ""
    
    syls_8 = eight_line.strip().split()
    if len(syls_8) >= 8:
        syl_8 = re.sub(r'[,.!?;:]+$', '', syls_8[7])  # strip trailing punctuation
        rhyme = get_rhyme_group(syl_8)  # pos 8 → chain rhyme
        rhyme_tag = f"[RHYME:{rhyme}]"
    
    syls_6 = six_line.strip().split()
    if len(syls_6) >= 6:
        seq = get_tone_sequence(six_line)
        tone_tag = f"[TONE:{seq[:6]}]"
    
    return rhyme_tag, tone_tag


def get_doi_tho_tags_tn(seven_line: str) -> tuple[str, str]:
    """
    Extract [RHYME:X] and [TONE:YYYYYYY] for Thất Ngôn đối thơ.
    
    [RHYME:X] — from position 7 (last syllable) of the input 7-syl line
    [TONE:YYYYYYY] — tone pattern of the 7-syl line (7 chars)
    
    Returns (rhyme_tag, tone_tag).
    """
    import re
    rhyme_tag = ""
    tone_tag = ""
    
    syls = seven_line.strip().split()
    if len(syls) >= 7:
        syl_7 = re.sub(r'[,.!?;:]+$', '', syls[6])  # pos 7, strip punctuation
        rhyme = get_rhyme_group(syl_7)
        rhyme_tag = f"[RHYME:{rhyme}]"
        seq = get_tone_sequence(seven_line)
        tone_tag = f"[TONE:{seq[:7]}]"  # 7-char tone sequence
    
    return rhyme_tag, tone_tag
