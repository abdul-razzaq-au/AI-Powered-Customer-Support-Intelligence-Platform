import pandas as pd


from model_loader import (
    priority_model,
    priority_label_encoder,
    le_gender,
    le_product,
    le_channel
)


from feature_engineering import (
    create_text_features,
    create_ticket_age,
    safe_label_encode
)



def prepare_priority_features(data):


    char_count, word_count, avg_word_length = (
        create_text_features(
            data["ticket_description"]
        )
    )


    ticket_age_days = create_ticket_age(
        data["date_of_purchase"]
    )



    features = {


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



    return pd.DataFrame(
        [features]
    )




def predict_priority(data):


    X = prepare_priority_features(
        data
    )


    prediction = priority_model.predict(
        X
    )


    priority_label = (
        priority_label_encoder
        .inverse_transform(
            prediction.astype(int)
        )[0]
    )


    return priority_label