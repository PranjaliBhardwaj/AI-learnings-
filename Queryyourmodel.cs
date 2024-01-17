//To query your model using an SDK, you first need to create your client. Once you have your client, you then use it to call the appropriate endpoint.
var languageClient = new TextAnalyticsClient(endpoint, credentials);
var response = languageClient.ExtractKeyPhrases(document);
//Other language features, such as the conversational language understanding, require the request be built and sent differently.
var data = new
{
    analysisInput = new
    {
        conversationItem = new
        {
            text = userText,
            id = "1",
            participantId = "1",
        }
    },
    parameters = new
    {
        projectName,
        deploymentName,
        // Use Utf16CodeUnit for strings in .NET.
        stringIndexType = "Utf16CodeUnit",
    },
    kind = "Conversation",
};
Response response = await client.AnalyzeConversationAsync(RequestContent.Create(data));
