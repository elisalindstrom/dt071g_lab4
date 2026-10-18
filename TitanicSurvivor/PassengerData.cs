// Klass för att läsa in och lägga till en passagerare

using Microsoft.ML.Data;

namespace titanicSurvivor
{
    // Kolumner i csv-fil: PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
    public class PassengerData
    {
        [LoadColumn(1)] public bool Survived { get; set; }
        [LoadColumn(2)] public float Pclass { get; set; }
        [LoadColumn(4)] public string Sex { get; set; } = "";
        [LoadColumn(5)] public float Age { get; set; }
        [LoadColumn(6)] public float SibSp { get; set; }
        [LoadColumn(7)] public float Parch { get; set; }
        [LoadColumn(10)] public string Cabin { get; set; } = "";
    }
}