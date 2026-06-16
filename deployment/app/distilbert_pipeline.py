import torch


from model_loader import (
    distilbert_model,
    distilbert_tokenizer,
    ticket_type_encoder
)



def predict_ticket_type(
        ticket_subject,
        ticket_description
):

    """
    Predict ticket category using DistilBERT
    """


    # Same text format used during training
    from text_preprocessing import (
    create_processed_combined_text
    )


    combined_text = create_processed_combined_text(
    ticket_subject,
    ticket_description
    )


    # Tokenization

    inputs = distilbert_tokenizer(
        combined_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )


    # Prediction

    with torch.no_grad():

        outputs = distilbert_model(
            **inputs
        )


    prediction_id = torch.argmax(
        outputs.logits,
        dim=1
    ).item()



    # Convert ID back to label

    ticket_type = (
        ticket_type_encoder
        .inverse_transform(
            [prediction_id]
        )[0]
    )


    return ticket_type