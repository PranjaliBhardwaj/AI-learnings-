//To query your model using an SDK, you first need to create your client. Once you have your client, you then use it to call the appropriate endpoint.
language_client = TextAnalyticsClient(
            endpoint=endpoint, 
            credential=credentials)
response = language_client.extract_key_phrases(documents = documents)[0]
//Other language features, such as the conversational language understanding, require the request be built and sent differently.
result = client.analyze_conversation(
    task={
        "kind": "Conversation",
        "analysisInput": {
            "conversationItem": {
                "participantId": "1",
                "id": "1",
                "modality": "text",
                "language": "en",
                "text": query
            },
            "isLoggingEnabled": False
        },
        "parameters": {
            "projectName": cls_project,
            "deploymentName": deployment_slot,
            "verbose": True
        }
    }
) 
