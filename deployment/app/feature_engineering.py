import pandas as pd
from datetime import datetime


# Same reference date used during training
# IMPORTANT:
# replace this with your actual training reference date

REFERENCE_DATE = pd.Timestamp(
    "2026-04-12"
)


def create_text_features(text):

    char_count = len(text)

    word_count = len(
        text.split()
    )

    avg_word_length = (
        char_count / word_count
        if word_count > 0
        else 0
    )

    return (
        char_count,
        word_count,
        avg_word_length
    )



def create_ticket_age(date_of_purchase):

    purchase_date = pd.to_datetime(
        date_of_purchase
    )


    ticket_age_days = (
        REFERENCE_DATE -
        purchase_date
    ).days


    return ticket_age_days

def safe_label_encode(
        encoder,
        value
):

    if value in encoder.classes_:

        return encoder.transform(
            [value]
        )[0]

    else:

        return -1