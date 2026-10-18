// Hanterar inläsning, träning, utvärdering och förutsägelser

using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace titanicSurvivor
{
    public class TitanicPredictor
    {
        MLContext mlContext = new MLContext(seed: 1);

        // Sökväg till csv-fil
        string csvPath = "Titanic-Dataset.csv";

        // Läser in data från csv-filen
        public IDataView Load()
        {
            // Kontroll om fil finns
            if (!File.Exists(csvPath))
            {
                throw new FileNotFoundException($"Kunde inte hitta filen: {Path.GetFullPath(csvPath)}");
            }

            // Läser in fil
            IDataView data = mlContext.Data.LoadFromTextFile<PassengerData>(
                csvPath,
                hasHeader: true,
                separatorChar: ',',
                allowQuoting: true
            );

            // Returnerar data från fil
            return data;
        }

        // Skapar en pipeline
        public IEstimator<ITransformer> CreatePipeline()
        {
            var pipeline =
                // Omvandlar data till passande format
                mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "SexEncoded",
                    inputColumnName: nameof(PassengerData.Sex))
                .Append(mlContext.Transforms.Categorical.OneHotEncoding(
                    outputColumnName: "CabinEncoded",
                    inputColumnName: nameof(PassengerData.Cabin)))
                // Om uppgift saknas ersätts värde med medelvärdet
                .Append(mlContext.Transforms.ReplaceMissingValues(
                    outputColumnName: "AgeFilled",
                    inputColumnName: nameof(PassengerData.Age),
                    replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                // Om uppgift saknas ersätts värdet med standardvärde = 0
                .Append(mlContext.Transforms.ReplaceMissingValues(
                    outputColumnName: "ParentChildFilled",
                    inputColumnName: nameof(PassengerData.Parch),
                    replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))
                .Append(mlContext.Transforms.ReplaceMissingValues(
                    outputColumnName: "PclassFilled",
                    inputColumnName: nameof(PassengerData.Pclass),
                    replacementMode: MissingValueReplacingEstimator.ReplacementMode.Mean))
                .Append(mlContext.Transforms.ReplaceMissingValues(
                    outputColumnName: "SiblingSpouseFilled",
                    inputColumnName: nameof(PassengerData.SibSp),
                    replacementMode: MissingValueReplacingEstimator.ReplacementMode.DefaultValue))
                .Append(mlContext.Transforms.Concatenate(
                    "Features",
                    "AgeFilled",
                    "SexEncoded",
                    "CabinEncoded",
                    "ParentChildFilled",
                    "PclassFilled",
                    "SiblingSpouseFilled"
            ))
            .Append(mlContext.Transforms.NormalizeMinMax("Features"))
            .Append(
                    mlContext.BinaryClassification.Trainers
                        .SdcaLogisticRegression(
                            labelColumnName: nameof(PassengerData.Survived),
                            featureColumnName: "Features"
                        )
                );
            // Returnerar pipeline
            return pipeline;
        }

        // Delar upp data, tränar och utvärderar modellen
        public CalibratedBinaryClassificationMetrics Evaluate()
        {
            // Läser in data och skapar pipeline
            IDataView data = Load();
            var pipeline = CreatePipeline();

            // Delar upp data i 80% träningsdata och 20% testdata
            var split = mlContext.Data.TrainTestSplit(
                data,
                testFraction: 0.2,
                seed: 1
            );

            // Tränar modellen på träningsdata
            var modelOnTrain = pipeline.Fit(split.TrainSet);

            // Använder modellen för att göra förutsägelser i testdata
            var testPredictions = modelOnTrain.Transform(split.TestSet);

            // Utvärderar modellen
            var testMetrics = mlContext.BinaryClassification.Evaluate(
                testPredictions,
                labelColumnName: nameof(PassengerData.Survived)
            );

            // Returnerar utvärdering
            return testMetrics;
        }

        // Beräknar förutsägelse om ny passagerares överlevnad
        public PassengerPrediction Predict(PassengerData newPassenger)
        {
            ITransformer model = Train();

            // Skapar en prediction engine
            var predictionEngine =
                mlContext.Model.CreatePredictionEngine<PassengerData, PassengerPrediction>(model);

            // Beräknar överlevnad
            PassengerPrediction prediction = predictionEngine.Predict(newPassenger);

            // Returnerar resultatet
            return prediction;
        }

        // Tränar modellen på all data för att kunna förutsäga överlevnad på nya passagerare
        public ITransformer Train()
        {
            IDataView data = Load();
            var pipeline = CreatePipeline();
            var model = pipeline.Fit(data);
            return model;
        }
    }
}