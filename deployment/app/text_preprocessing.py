import re
import string
import contractions

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



stop_words = set(
    stopwords.words('english')
)

lemmatizer = WordNetLemmatizer()



def clean_text(text):

    text = str(text).lower()

    # Remove HTML tags
    text = re.sub(
        r'<.*?>',
        '',
        text
    )

    # Remove contractions
    text = contractions.fix(text)

    # Remove placeholders
    text = re.sub(
        r'\{.*?\}',
        ' ',
        text
    )

    # Remove URLs
    text = re.sub(
        r'http\S+|www\S+',
        '',
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            '',
            '',
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r'\s+',
        ' ',
        text
    ).strip()

    return text



def preprocess_text(text):

    text = text.lower()

    words = text.split()

    words = [
        word
        for word in words
        if word not in stop_words
    ]

    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]

    return " ".join(words)



def create_processed_combined_text(
        ticket_subject,
        ticket_description
):

    clean_subject = clean_text(
        ticket_subject
    )

    clean_description = clean_text(
        ticket_description
    )


    processed_subject = preprocess_text(
        clean_subject
    )

    processed_description = preprocess_text(
        clean_description
    )


    combined_text = (
        processed_subject
        + " "
        + processed_description
    )


    return combined_text