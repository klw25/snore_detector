//
//  SnoreModel.swift
//  SnoreDetector
//
//  Created by Kenny Withers on 12/19/25.
//


import CoreML

/*final class SnoreModel {
    private let model: MLModel

    init() {
        let config = MLModelConfiguration()
        self.model = try! snore_cnn(configuration: config).model
    }

    func predict(mel: MLMultiArray) -> Double {
        let input = try! MLDictionaryFeatureProvider(dictionary: [
            "input_audio": MLFeatureValue(multiArray: mel)
        ])

        let output = try! model.prediction(from: input)
        return output.featureValue(for: "var_102")!.doubleValue
    }
}*/


final class SnoreModel {
    private let model: MLModel

    init() {
        let config = MLModelConfiguration()
        // Ensure 'snore_cnn' matches your Xcode generated class name exactly
        do {
            self.model = try snore_cnn(configuration: config).model
        } catch {
            fatalError("❌ Failed to load model: \(error)")
        }
    }

    /*func predict(mel: MLMultiArray?) -> Double {
        // 1. Check if MelSpectrogram generation succeeded
        guard let mel = mel else {
            print("⚠️ Skipping prediction: Invalid Mel Spectrogram")
            return 0.0
        }

        do {
            let input = try MLDictionaryFeatureProvider(dictionary: [
                "input_audio": MLFeatureValue(multiArray: mel)
            ])

            let output = try model.prediction(from: input)
            
            // 2. Safe return of the specific output variable
            // Double check "var_102" is the correct name in your mlpackage!
            return output.featureValue(for: "var_102")?.doubleValue ?? 0.0
            
        } catch {
            print("❌ Model Prediction Error: \(error)")
            return 0.0
        }
    }*/
    func predict(mel: MLMultiArray?) -> Double {
            guard let mel = mel else { return 0.0 }

            do {
                let input = try MLDictionaryFeatureProvider(dictionary: [
                    "input_audio": MLFeatureValue(multiArray: mel)
                ])

                let output = try model.prediction(from: input)
                
                // 1️⃣ Grab the Raw Output Array (Logits)
                // It looks like: [Non-Snore-Score, Snore-Score] -> e.g. [-2.0, 4.5]
                guard let rawOutput = output.featureValue(for: "var_102")?.multiArrayValue else {
                    return 0.0
                }
                
                // 2️⃣ Extract the two scores
                // Note: MLMultiArray is annoying to read, we cast to NSNumber first
                let nonSnoreScore = rawOutput[0].doubleValue
                let snoreScore = rawOutput[1].doubleValue
                
                // 3️⃣ Math: Softmax (Convert Raw Scores to 0-1 Probability)
                // Formula: e^snore / (e^nonSnore + e^snore)
                let expSnore = exp(snoreScore)
                let expNonSnore = exp(nonSnoreScore)
                let snoreProbability = expSnore / (expNonSnore + expSnore)
                
                return snoreProbability
                
            } catch {
                print("❌ Model Prediction Error: \(error)")
                return 0.0
            }
        }
}
