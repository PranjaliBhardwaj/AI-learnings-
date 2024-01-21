using Azure.AI.Vision.Common;
using Azure.AI.Vision.ImageAnalysis;

var serviceOptions = new VisionServiceOptions(
    Environment.GetEnvironmentVariable("VISION_ENDPOINT"),
    new AzureKeyCredential(Environment.GetEnvironmentVariable("VISION_KEY")));

using var imageSource = VisionSource.FromUrl(new Uri("<url>"));

var options = new ImageAnalysisOptions()
{
    Features = ImageAnalysisFeature.Caption | ImageAnalysisFeature.Text,
    Language = "en",
    GenderNeutralCaption = true
};

using var analyzer = new ImageAnalyzer(serviceOptions, imageSource, options);

var result = analyzer.Analyze();




# To make an OCR REQUEST TO IMAGE ANANLYSIS
var analysisOptions = new ImageAnalysisOptions()
{
    Features = ImageAnalysisFeature.Text,
};
