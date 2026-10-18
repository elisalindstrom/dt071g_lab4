// Klass för modellens gissning om en passagerares överlevnad

using Microsoft.ML.Data;

namespace titanicSurvivor
{
    public class PassengerPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Survived { get; set; }

        public float Probability { get; set; }

        public float Score { get; set; }
    }
}