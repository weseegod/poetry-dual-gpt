# data/preprocess.py
# ===================
# Purpose: Convert raw Vietnamese poetry text into training-ready format.
#
# Input:  Raw poetry corpus (one poem per stanza, lines separated by newlines,
#         stanzas separated by blank lines)
# Output: One training example per line, with control tokens:
#         <|start|> [LUC_BAT] Line 1, <|reply|> Line 2 <|end|>
#
# What you'll learn:
#   - How to structure text data for causal language modeling
#   - What control tokens are and why they matter for conditional generation
#   - Vietnamese poetic forms: Lục Bát (6-8), Tứ Tuyệt (7-7-7-7),
#     Thất Ngôn Bát Cú (7-7-7-7-7-7-7-7)
#
# Key concepts:
#   - "Causal LM" predicts next token given previous tokens
#   - Each line becomes: [prompt] → [reply] pair
#   - Genre tags like [LUC_BAT] act as "instructions" to the model
#
# Implementation plan:
#   1. Read raw corpus file (blank-line separated stanzas)
#   2. Detect poetic genre from syllable pattern
#      - Lục Bát: lines alternate 6-8-6-8...
#      - Tứ Tuyệt: 4 lines of 7 syllables each
#      - Thất Ngôn Bát Cú: 8 lines of 7 syllables each
#   3. Split each stanza into (prompt_line, reply_line) pairs
#   4. Wrap each pair with control tokens
#   5. Write to output file, one pair per line
#
# Vietnamese syllable counting tip:
#   Count groups of consecutive letters + diacritics separated by spaces
#   Example: "Trăm năm trong cõi người ta" = 6 syllables

# --- YOUR CODE BELOW ---
