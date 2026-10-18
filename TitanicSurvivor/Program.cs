/*
Laboration 4 kurs DT071G

Program som kan förutsäga om en passagerare har möjlighet att överleva Titanic.
Användaren kan mata in uppgifter om en passagerare och beroende på bland annat ålder, kön och biljettklass kommer förutsägelsen förändras.
Programmet visar även en utvärdering av modellen som används.

Av Elisa Lindström 2026
*/

namespace titanicSurvivor
{
    public class Program
    {
        static void Main(string[] args)
        {
            // Skapar instans för klass som hanterar modell
            TitanicPredictor titanicPredictor = new TitanicPredictor();

            // Anropar metod som används för att dela upp och träna/utvärdera data
            var testMetrics = titanicPredictor.Evaluate();

            // Utvärdering av modellen
            Console.WriteLine("UTVÄRDERING AV SPLIT MED 80% TRÄNINGSDATA och 20% TESTDATA");
            Console.WriteLine($"Accuracy: {testMetrics.Accuracy:P2}");
            Console.WriteLine($"AUC: {testMetrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"F1 score: {testMetrics.F1Score:P2}");

            // Confusion matrix som visar hur väl modellens förutsägelser om överlevnad stämmer överens med testdata
            Console.WriteLine("\nMODELLENS FÖRUTSÄGELSER JÄMFÖRT MED TESTDATA:");
            Console.WriteLine(testMetrics.ConfusionMatrix.GetFormattedConfusionTable());

            // Deklarerar variabler för input
            string? ageInput;
            string? sexInput;
            string? classInput;
            string? cabinInput;
            string? parchInput;
            string? sibSpInput;

            // Variabler efter validering
            float age;
            float pclass;
            float parch;
            float sibSp;

            // Frågar användaren efter passagerarens uppgifter
            Console.WriteLine("\nLÄGG TILL NY PASSAGERARE");

            // Validering av input och format
            Console.Write("Ange passagerarens ålder: ");
            ageInput = Console.ReadLine();
            while (!float.TryParse(ageInput, out age))
            {
                Console.WriteLine("Felaktigt värde, ange passagerarens ålder i siffror: ");
                ageInput = Console.ReadLine();
            }

            Console.Write("Ange passagerarens kön (male/female): ");
            sexInput = Console.ReadLine();
            while (sexInput != "male" && sexInput != "female")
            {
                Console.WriteLine("Felaktigt värde, ange kön som male eller female: ");
                sexInput = Console.ReadLine();
            }

            Console.Write("Ange passagerarens biljettklass (1-3): ");
            classInput = Console.ReadLine();
            while (!float.TryParse(classInput, out pclass) || pclass < 1 || pclass > 3)
            {
                Console.WriteLine("Felaktig biljettklass, ange biljettklass 1, 2 eller 3: ");
                classInput = Console.ReadLine();
            }

            Console.Write("Ange passagerarens hyttkategori (lämna tomt om okänd): ");
            cabinInput = Console.ReadLine();

            Console.Write("Ange passagerarens antal barn eller föräldrar ombord: ");
            parchInput = Console.ReadLine();
            while (!float.TryParse(parchInput, out parch))
            {
                Console.WriteLine("Felaktigt värde, ange ett heltal: ");
                parchInput = Console.ReadLine();
            }

            Console.Write("Ange passagerarens antal syskon eller make/maka ombord: ");
            sibSpInput = Console.ReadLine();
            while (!float.TryParse(sibSpInput, out sibSp))
            {
                Console.WriteLine("Felaktigt värde, ange ett heltal: ");
                sibSpInput = Console.ReadLine();
            }

            // Skapar passagerarobjekt med uppgifterna
            PassengerData newPassenger = new()
            {
                Age = age,
                Sex = sexInput,
                Pclass = pclass,
                Cabin = cabinInput ?? "",
                Parch = parch,
                SibSp = sibSp
            };

            // Använder metoden Predict för att beräkna förutsägelse om passagerarens överlevnad
            PassengerPrediction prediction = titanicPredictor.Predict(newPassenger);

            Console.WriteLine(
                prediction.Survived
                    ? "Modellen tror att passageraren överlever."
                    : "Modellen tror att passageraren inte överlever."
            );

            Console.WriteLine($"Sannolikhet: {prediction.Probability:P1}");
        }
    }
}

