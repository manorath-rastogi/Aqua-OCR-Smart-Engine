"""
AQIVON OmniOCR Benchmark & Evaluation Metrics.
Provides Character Error Rate (CER), Word Error Rate (WER), Levenshtein edit distance,
and throughput tracking for OCR performance verification.
"""

from typing import Dict, List, Tuple


class OCRMetrics:
    """Calculates accuracy and distance metrics between reference text and OCR hypothesis."""

    @staticmethod
    def levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
        """Computes Wagner-Fischer dynamic programming edit distance."""
        len1, len2 = len(seq1), len(seq2)
        dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            dp[i][0] = i
        for j in range(len2 + 1):
            dp[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,       # Deletion
                    dp[i][j - 1] + 1,       # Insertion
                    dp[i - 1][j - 1] + cost # Substitution
                )

        return dp[len1][len2]

    @staticmethod
    def compute_cer(reference: str, hypothesis: str) -> float:
        """
        Computes Character Error Rate (CER).
        Returns float between 0.0 (perfect) and 1.0+.
        """
        ref_chars = list(reference)
        hyp_chars = list(hypothesis)
        if not ref_chars:
            return 0.0 if not hyp_chars else 1.0

        dist = OCRMetrics.levenshtein_distance(ref_chars, hyp_chars)
        return float(dist) / float(len(ref_chars))

    @staticmethod
    def compute_wer(reference: str, hypothesis: str) -> float:
        """
        Computes Word Error Rate (WER).
        Returns float between 0.0 (perfect) and 1.0+.
        """
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        if not ref_words:
            return 0.0 if not hyp_words else 1.0

        dist = OCRMetrics.levenshtein_distance(ref_words, hyp_words)
        return float(dist) / float(len(ref_words))

    @staticmethod
    def compute_benchmark_summary(results: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Computes average CER, WER, and accuracy across a list of reference-prediction pairs.
        Each item is {'reference': str, 'prediction': str}.
        """
        if not results:
            return {"cer": 0.0, "wer": 0.0, "character_accuracy": 100.0, "word_accuracy": 100.0}

        cers = [OCRMetrics.compute_cer(item["reference"], item["prediction"]) for item in results]
        wers = [OCRMetrics.compute_wer(item["reference"], item["prediction"]) for item in results]

        avg_cer = sum(cers) / len(cers)
        avg_wer = sum(wers) / len(wers)

        return {
            "avg_cer": round(avg_cer, 4),
            "avg_wer": round(avg_wer, 4),
            "char_accuracy_pct": round(max(0.0, (1.0 - avg_cer) * 100.0), 2),
            "word_accuracy_pct": round(max(0.0, (1.0 - avg_wer) * 100.0), 2)
        }

