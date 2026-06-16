import pandas as pd

from scipy.sparse import hstack


from model_loader import (
    regression_model,
    tfidf_vectorizer,
    le_gender,
    le_product,
    le_channel
)


from feature_engineering import (
    create_text_features,
    create_ticket_age,
    safe_label_encode
)




def prepare_regression_features(data):


    char_count, word_count, avg_word_length = (
        create_text_features(
            data["ticket_description"]
        )
    )


    ticket_age_days = create_ticket_age(
        data["date_of_purchase"]
    )



    tabular_features = {


        "Customer Age":
            data["customer_age"],


        "Customer Gender Encoded":
            safe_label_encode(
                le_gender,
                data["customer_gender"]
            ),


        "Product Purchased Encoded":
            safe_label_encode(
                le_product,
                data["product_purchased"]
            ),


        "Ticket Channel Encoded":
            safe_label_encode(
                le_channel,
                data["ticket_channel"]
            ),


        "ticket_age_days":
            ticket_age_days,


        "char_count":
            char_count,


        "word_count":
            word_count,


        "avg_word_length":
            avg_word_length

    }



    tabular_df = pd.DataFrame(
        [tabular_features]
    )



    # Same text used during training

    from text_preprocessing import preprocess_text


    combined_text = (
        data["ticket_subject"]
        +
        " "
        +
        data["ticket_description"]
    )


    combined_text = preprocess_text(
        combined_text
    )

    text_features = tfidf_vectorizer.transform(
        [combined_text]
    )

    final_features = hstack(
        [
            tabular_df,
            text_features
        ]
    )


    return final_features





def predict_resolution_time(data):


    X = prepare_regression_features(
        data
    )


    prediction = regression_model.predict(
        X
    )[0]


    return float(prediction)