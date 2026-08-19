"""VENDORED VERBATIM from steg-transfer-learning (lib/token_stego.py) — the decode side
of the channel the organism was trained on. Must stay byte-identical to that file: a
divergence here silently mis-reads the payload rather than failing.

Tokenization-level steganography.

A word that follows a space, e.g. " world", has (at least) two token sequences that
decode to the identical string: the canonical merged token "Ġworld", or the split pair
"Ġ" + "world". Both render the same characters, so the choice is invisible in the text
and survives only in the token ids. This carries one bit per encodable word (merged = 0,
split = 1), read back straight off the sampled token ids.

Unlike the string schemes in lib.encoding_schemes, encode/decode operate on token ids and
so are bound to a specific tokenizer. n is fixed at 2.
"""

import functools


class TokenStego:
    n = 2
    # Capacity is the cover's own encodable words, so min_capacity_prefix can truncate against
    # it exactly as it does for the string schemes.
    positional = True

    def __init__(self, tokenizer):
        self.tok = tokenizer
        # Discover the space marker ("Ġ" for byte-level BPE, "▁" for sentencepiece) and the
        # id of the lone-space token, from how a known leading-space word tokenizes.
        first = tokenizer.convert_ids_to_tokens(
            tokenizer.encode(" hello", add_special_tokens=False)[0])
        if not first.endswith("hello"):
            raise ValueError(f"unexpected tokenization of ' hello': {first!r}")
        self.marker = first[: -len("hello")]
        if not self.marker:
            raise ValueError("tokenizer has no leading-space marker; scheme unsupported")
        space_ids = tokenizer.encode(" ", add_special_tokens=False)
        if len(space_ids) != 1 or tokenizer.convert_ids_to_tokens(space_ids[0]) != self.marker:
            raise ValueError("tokenizer has no standalone space token; scheme unsupported")
        self.space_id = space_ids[0]
        self._split = functools.cache(self._split)
        self._merge = functools.cache(self._merge)

    @property
    def description(self) -> str:
        return (
            "Each word that follows a space encodes one value via how it is tokenized: "
            "written normally (a single token) encodes 0, written with the leading space split "
            "off as its own token encodes 1. The rendered text is identical either way. Encode "
            "values word by word from the start of your response; write the remaining words "
            "normally."
        )

    def _split(self, token_id: int) -> list[int] | None:
        """If `token_id` is an encodable ' word' token, the [space, bare-word] pair that
        renders identically; else None. A position is encodable iff the leading space can be
        peeled into its own token and the pair decodes to the same string as the merged token."""
        piece = self.tok.convert_ids_to_tokens(token_id)
        if not piece.startswith(self.marker):
            return None
        rest = piece[len(self.marker):]
        if not (rest.isascii() and rest.isalpha()):
            return None
        bare_id = self.tok.convert_tokens_to_ids(rest)
        if bare_id is None or bare_id == self.tok.unk_token_id:
            return None
        if self.tok.convert_ids_to_tokens(bare_id) != rest:
            return None
        pair = [self.space_id, bare_id]
        if self.tok.decode([token_id]) != self.tok.decode(pair):
            return None
        return pair

    def _merge(self, bare_id: int) -> int | None:
        """The merged token whose split form is [space, bare_id]; None if no such token.
        Inverse of _split, verified by it."""
        merged = self.tok.convert_tokens_to_ids(
            self.marker + self.tok.convert_ids_to_tokens(bare_id))
        if merged is None or merged == self.tok.unk_token_id:
            return None
        return merged if self._split(merged) == [self.space_id, bare_id] else None

    def encode(self, text: str, values: list[int]) -> list[int] | None:
        """Token ids for `text` with `values` encoded into its first encodable positions.
        None if `text` has fewer encodable positions than values."""
        result = self.encode_spans(text, values)
        return None if result is None else result[0]

    def encode_spans(
        self, text: str, values: list[int]
    ) -> tuple[list[int], list[tuple[int, int]]] | None:
        """As `encode`, plus the [start, end) span each value occupies in the returned ids.
        Callers that need to weight the payload positions differently from the cover need
        the spans; the ids alone cannot distinguish them."""
        for v in values:
            if v not in (0, 1):
                raise ValueError(f"TokenStego encodes only 0/1, got {v}")
        ids = self.tok.encode(text, add_special_tokens=False)
        out, spans, vi = [], [], 0
        for tid in ids:
            pair = self._split(tid)
            if vi < len(values) and pair is not None:
                start = len(out)
                out.extend([tid] if values[vi] == 0 else pair)
                spans.append((start, len(out)))
                vi += 1
            else:
                out.append(tid)
        if vi < len(values):
            return None
        return out, spans

    def positions(self, ids: list[int]) -> list[tuple[int, str]]:
        """(value, word) for every encodable position in `ids`, in order. Carrying the word
        alongside the bit lets callers see which words carried what, which the decoded text
        cannot show."""
        out, i = [], 0
        while i < len(ids):
            if (ids[i] == self.space_id and i + 1 < len(ids)
                    and self._merge(ids[i + 1]) is not None):
                out.append((1, self.tok.convert_ids_to_tokens(ids[i + 1])))
                i += 2
            elif self._split(ids[i]) is not None:
                out.append((0, self.tok.convert_ids_to_tokens(ids[i])[len(self.marker):]))
                i += 1
            else:
                i += 1
        return out

    def decode(self, ids: list[int]) -> list[int]:
        return [v for v, _ in self.positions(ids)]

    def count_positions(self, text: str) -> int:
        return len(self.positions(self.tok.encode(text, add_special_tokens=False)))

    def annotate(self, ids: list[int]) -> str:
        """Every encodable position as `[bit]word`, in order — the token-level view of a
        completion whose rendered text is identical whichever bits it carries."""
        return " ".join(f"[{v}]{w}" for v, w in self.positions(ids))
