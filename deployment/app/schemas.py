from pydantic import BaseModel


class TicketRequest(BaseModel):

    # Text inputs
    ticket_subject: str
    ticket_description: str

    # Structured inputs
    customer_age: int
    customer_gender: str
    product_purchased: str
    ticket_channel: str

    # Date feature
    date_of_purchase: str



class PredictionResponse(BaseModel):

    ticket_type: str

    ticket_priority: str

    estimated_resolution_hours: float