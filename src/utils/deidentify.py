import re

class TextDeidentifier:
    """
    De-identification utility to scrub PII and platform-specific identifiers
    from social media text before processing, adhering to ethical guidelines.
    """
    URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
    USER_MENTION_PATTERN = re.compile(r'(?:@|u/|/u/)[A-Za-z0-9_-]+')
    SUBREDDIT_PATTERN = re.compile(r'(?:r/|/r/)[A-Za-z0-9_-]+')
    EMAIL_PATTERN = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
    PHONE_PATTERN = re.compile(r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}')

    def __init__(self, mask_tokens: bool = True):
        self.mask_tokens = mask_tokens

    def clean(self, text: str) -> str:
        if not text or not isinstance(text, str):
            return ""

        # Replace Emails first to prevent @domain from matching user mentions
        rep_email = "[EMAIL]" if self.mask_tokens else ""
        text = self.EMAIL_PATTERN.sub(rep_email, text)

        # Replace URLs
        rep_url = "[URL]" if self.mask_tokens else ""
        text = self.URL_PATTERN.sub(rep_url, text)

        # Replace Subreddit / Community tags
        rep_sub = "[COMMUNITY]" if self.mask_tokens else ""
        text = self.SUBREDDIT_PATTERN.sub(rep_sub, text)

        # Replace User mentions
        rep_user = "[USER]" if self.mask_tokens else ""
        text = self.USER_MENTION_PATTERN.sub(rep_user, text)

        # Replace Phone numbers
        rep_phone = "[PHONE]" if self.mask_tokens else ""
        text = self.PHONE_PATTERN.sub(rep_phone, text)

        # Clean excess whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
