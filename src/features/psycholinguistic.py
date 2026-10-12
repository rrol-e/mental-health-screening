import re
import math
from typing import Dict, Any, List

class PsycholinguisticExtractor:
    """
    Extracts psycholinguistic and computational psychiatry feature vectors
    from de-identified text based on established research literature:
    - 1st-person singular pronoun density (self-referential focus)
    - Absolutist word frequency (cognitive distortion / black-and-white thinking)
    - Negative emotion and distress indicators
    - Punctuation and syntactic hesitation markers (ellipses, question density)
    """

    FIRST_PERSON_SINGULAR = {
        'i', 'me', 'my', 'mine', 'myself'
    }
    
    FIRST_PERSON_PLURAL = {
        'we', 'us', 'our', 'ours', 'ourselves'
    }

    SECOND_THIRD_PERSON = {
        'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers', 
        'they', 'them', 'their', 'theirs'
    }

    # Established Absolutist Dictionary (Al-Mosaiwi & Johnstone, 2018)
    ABSOLUTIST_WORDS = {
        'absolutely', 'all', 'always', 'complete', 'completely', 'constant',
        'constantly', 'definitely', 'entire', 'entirely', 'ever', 'every',
        'everyone', 'everything', 'full', 'fully', 'must', 'never', 'no one',
        'nobody', 'none', 'nothing', 'totally', 'whole', 'unbearable', 'impossible'
    }

    # Lexicon for distress, anhedonia, and negative affective state
    DISTRESS_AFFECT_WORDS = {
        'sad', 'hopeless', 'worthless', 'guilt', 'guilty', 'tired', 'exhausted',
        'empty', 'hollow', 'alone', 'lonely', 'isolated', 'pain', 'hurts', 'hurting',
        'miserable', 'numb', 'failure', 'fail', 'failed', 'cry', 'crying', 'tears',
        'anxious', 'anxiety', 'panic', 'scared', 'afraid', 'drowning', 'suffocating',
        'helpless', 'burden', 'hate', 'hating', 'broken', 'lost', 'struggle', 'struggling'
    }

    # Sleep and somatic disruption markers
    SOMATIC_WORDS = {
        'sleep', 'sleeping', 'insomnia', 'awake', 'nightmare', 'nightmares',
        'bed', 'headache', 'stomach', 'fatigue', 'drowsy', 'appetite', 'eat', 'eating'
    }

    def __init__(self):
        self.word_token_pattern = re.compile(r"\b[a-zA-Z']+\b")

    def tokenize(self, text: str) -> List[str]:
        return self.word_token_pattern.findall(text.lower())

    def extract_features(self, text: str) -> Dict[str, float]:
        tokens = self.tokenize(text)
        total_words = len(tokens)
        
        if total_words == 0:
            return {
                "word_count": 0.0,
                "first_person_singular_ratio": 0.0,
                "first_person_plural_ratio": 0.0,
                "self_focus_index": 0.0,
                "absolutist_ratio": 0.0,
                "distress_affect_ratio": 0.0,
                "somatic_ratio": 0.0,
                "lexical_diversity": 0.0,
                "avg_word_length": 0.0,
                "ellipsis_frequency": 0.0,
                "question_density": 0.0,
                "exclamation_density": 0.0,
            }

        # Count categories
        fp_sing_count = sum(1 for w in tokens if w in self.FIRST_PERSON_SINGULAR)
        fp_plur_count = sum(1 for w in tokens if w in self.FIRST_PERSON_PLURAL)
        other_pron_count = sum(1 for w in tokens if w in self.SECOND_THIRD_PERSON)
        absolutist_count = sum(1 for w in tokens if w in self.ABSOLUTIST_WORDS)
        distress_count = sum(1 for w in tokens if w in self.DISTRESS_AFFECT_WORDS)
        somatic_count = sum(1 for w in tokens if w in self.SOMATIC_WORDS)

        # Ratios normalized by total words
        fp_sing_ratio = fp_sing_count / total_words
        fp_plur_ratio = fp_plur_count / total_words
        
        # Self-focus index: 1st person singular relative to all pronouns
        total_pronouns = fp_sing_count + fp_plur_count + other_pron_count
        self_focus_index = (fp_sing_count / total_pronouns) if total_pronouns > 0 else 0.0

        absolutist_ratio = absolutist_count / total_words
        distress_affect_ratio = distress_count / total_words
        somatic_ratio = somatic_count / total_words

        # Lexical diversity (Type-Token Ratio)
        unique_tokens = len(set(tokens))
        lexical_diversity = unique_tokens / total_words

        # Average word length
        avg_word_length = sum(len(w) for w in tokens) / total_words

        # Punctuation markers
        ellipsis_count = len(re.findall(r'\.{2,}', text))
        question_count = text.count('?')
        exclamation_count = text.count('!')

        return {
            "word_count": float(total_words),
            "first_person_singular_ratio": round(fp_sing_ratio, 4),
            "first_person_plural_ratio": round(fp_plur_ratio, 4),
            "self_focus_index": round(self_focus_index, 4),
            "absolutist_ratio": round(absolutist_ratio, 4),
            "distress_affect_ratio": round(distress_affect_ratio, 4),
            "somatic_ratio": round(somatic_ratio, 4),
            "lexical_diversity": round(lexical_diversity, 4),
            "avg_word_length": round(avg_word_length, 2),
            "ellipsis_frequency": round(ellipsis_count / max(1, total_words / 50), 4),
            "question_density": round(question_count / max(1, total_words / 50), 4),
            "exclamation_density": round(exclamation_count / max(1, total_words / 50), 4),
        }
