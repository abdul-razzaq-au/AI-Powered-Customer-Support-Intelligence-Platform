from fastapi import FastAPI


from schemas import (
    TicketRequest,
    PredictionResponse
)


from distilbert_pipeline import (
    predict_ticket_type
)


from priority_pipeline import (
    predict_priority
)


from regression_pipeline import (
    predict_resolution_time
)



app = FastAPI(

    title="AI Powered Customer Support Intelligence Platform",

    description=(
        "API for ticket classification, "
        "priority prediction and resolution "
        "time estimation"
    ),

    version="1.0"
)



@app.get("/")
def home():

    return {

        "message":
        "Customer Support Intelligence API is running"

    }



@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: TicketRequest
):


    # Convert Pydantic object to dictionary

    data = request.model_dump()



    # -------------------------
    # Ticket Type Prediction
    # -------------------------

    ticket_type = predict_ticket_type(

        request.ticket_subject,

        request.ticket_description

    )



    # -------------------------
    # Priority Prediction
    # -------------------------

    priority = predict_priority(

        data

    )



    # -------------------------
    # Resolution Prediction
    # -------------------------

    resolution_hours = predict_resolution_time(

        data

    )



    return {


        "ticket_type":

            ticket_type,


        "ticket_priority":

            priority,


        "estimated_resolution_hours":

            round(
                resolution_hours,
                2
            )

    }