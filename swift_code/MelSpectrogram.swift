import Accelerate
import CoreML

enum MelSpectrogram {

    static let nFFT = 1024
    static let hop = 256
    static let nMels = 64
    static let frames = 63
    static let sampleRate: Float = 16_000
    
    static func compute(audio: [Float]) -> MLMultiArray? { // Return Optional to be safe

        // 1️⃣ CHANGE: Update shape to Rank 4 [Batch, Channel, Height, Width]
        let shape = [1, 1, nMels, frames].map { NSNumber(value: $0) }
            
        guard let mel = try? MLMultiArray(shape: shape, dataType: .float32) else {
            print("❌ Failed to create MLMultiArray")
            return nil
        }

        // 2️⃣ Hann window (Same as before)
        var window = [Float](repeating: 0, count: nFFT)
        vDSP_hann_window(&window, vDSP_Length(nFFT), Int32(vDSP_HANN_NORM))

        // 3️⃣ STFT magnitude (Same as before)
        let spectrogram = stftMagnitude(audio: audio, window: window)

        // 4️⃣ Mel filterbank (Same as before)
        let melFiltered = applyMelFilter(spectrogram: spectrogram)

        // 5️⃣ Log scaling
        /*for m in 0..<nMels {
            for t in 0..<frames {
                let v = max(melFiltered[m][t], 1e-10)
                    
                // 6️⃣ CHANGE: Access with 4 indices [Batch, Channel, Mel, Time]
                let index = [0, 0, m, t] as [NSNumber]
                mel[index] = NSNumber(value: logf(v))
            }
        }
            
            return mel
        }*/
        
        // ... Inside MelSpectrogram.compute ...

                // 5️⃣ Log Scaling & Fixed Normalization
                // We assume standard audio range: -80dB (Silence) to 0dB (Max Volume)
                let minDb: Float = -80.0
                let maxDb: Float = 0.0
                let dbRange = maxDb - minDb
                
                for m in 0..<nMels {
                    for t in 0..<frames {
                        let v = max(melFiltered[m][t], 1e-10)
                        
                        // 1. Convert to Log (Decibels)
                        // Note: log10 is more standard for dB, but let's stick to logf (natural log)
                        // if your model was trained that way.
                        // If this fails, try: 10 * log10f(v)
                        var db = logf(v)
                        
                        // 2. Clamp to our expected range
                        if db < minDb { db = minDb }
                        if db > maxDb { db = maxDb }
                        
                        // 3. Normalize (0.0 to 1.0)
                        // -80 becomes 0.0
                        // 0 becomes 1.0
                        let normalized = (db - minDb) / dbRange
                        
                        let index = [0, 0, m, t] as [NSNumber]
                        mel[index] = NSNumber(value: normalized)
                    }
                }
                
                return mel
            }
    /*static func compute(audio: [Float]) -> MLMultiArray {

        let mel = try! MLMultiArray(
            shape: [1, nMels, frames].map { NSNumber(value: $0) },
            dataType: .float32
        )

        // 1️⃣ Hann window
        var window = [Float](repeating: 0, count: nFFT)
        vDSP_hann_window(&window, vDSP_Length(nFFT), Int32(vDSP_HANN_NORM))

        // 2️⃣ STFT magnitude
        let spectrogram = stftMagnitude(audio: audio, window: window)

        // 3️⃣ Mel filterbank
        let melFiltered = applyMelFilter(spectrogram: spectrogram)

        // 4️⃣ Log scaling (torch-style)
        for m in 0..<nMels {
            for t in 0..<frames {
                let v = max(melFiltered[m][t], 1e-10)
                mel[[0, m, t] as [NSNumber]] = NSNumber(value: logf(v))
            }
        }
        
        //TEMPORARY
        // Sanity prints
        /*print("Mel shape:", mel.shape)

        print("Sample values:")
        print("mel[0,0,0] =", mel[[0, 0, 0]])
        print("mel[0,10,10] =", mel[[0, 10, 10]])
        print("mel[0,63,62] =", mel[[0, 63, 62]])*/


        return mel
    }*/

    // MARK: - Helpers

    private static func stftMagnitude(
        audio: [Float],
        window: [Float]
    ) -> [[Float]] {

        let fftSize = nFFT
        let halfFFT = fftSize / 2
        let frameCount = frames

        var result = Array(
            repeating: Array(repeating: Float(0), count: frameCount),
            count: halfFFT
        )

        var fftSetup = vDSP_create_fftsetup(
            vDSP_Length(log2(Float(fftSize))),
            FFTRadix(kFFTRadix2)
        )!

        for frame in 0..<frameCount {
            let start = frame * hop
            //let frameData = Array(audio[start..<start+fftSize])
            let end = start + fftSize
            var frameData: [Float]

            if end <= audio.count {
                // 1. Safe case: We have enough data
                frameData = Array(audio[start..<end])
            } else {
                // 2. Edge case: We are running off the end of the array
                let elementsAvailable = max(0, audio.count - start)
                
                if elementsAvailable > 0 {
                    // Grab what is left
                    frameData = Array(audio[start..<start+elementsAvailable])
                    // Fill the rest with zeros
                    let padding = [Float](repeating: 0, count: fftSize - elementsAvailable)
                    frameData.append(contentsOf: padding)
                } else {
                    // Start is already past the end (shouldn't happen if loop is correct, but safe to handle)
                    frameData = [Float](repeating: 0, count: fftSize)
                }
            }
            
            

            var windowed = zip(frameData, window).map(*)

            var real = windowed
            var imag = [Float](repeating: 0, count: fftSize)

            real.withUnsafeMutableBufferPointer { realPtr in
                imag.withUnsafeMutableBufferPointer { imagPtr in
                    var split = DSPSplitComplex(
                        realp: realPtr.baseAddress!,
                        imagp: imagPtr.baseAddress!
                    )

                    vDSP_fft_zrip(
                        fftSetup,
                        &split,
                        1,
                        vDSP_Length(log2(Float(fftSize))),
                        FFTDirection(FFT_FORWARD)
                    )

                    for i in 0..<halfFFT {
                        let r = split.realp[i]
                        let im = split.imagp[i]
                        result[i][frame] = sqrt(r*r + im*im)
                    }
                }
            }
        }

        vDSP_destroy_fftsetup(fftSetup)
        return result
    }

    private static func applyMelFilter(
        spectrogram: [[Float]]
    ) -> [[Float]] {

        let melBank = createMelFilterbank()

        var mel = Array(
            repeating: Array(repeating: Float(0), count: frames),
            count: nMels
        )

        for m in 0..<nMels {
            for f in 0..<spectrogram.count {
                let weight = melBank[m][f]
                for t in 0..<frames {
                    mel[m][t] += spectrogram[f][t] * weight
                }
            }
        }

        return mel
    }

    private static func createMelFilterbank() -> [[Float]] {

        let fftBins = nFFT / 2
        var filterbank = Array(
            repeating: Array(repeating: Float(0), count: fftBins),
            count: nMels
        )

        func hzToMel(_ hz: Float) -> Float {
            2595 * log10(1 + hz / 700)
        }

        func melToHz(_ mel: Float) -> Float {
            700 * (pow(10, mel / 2595) - 1)
        }

        let melMin = hzToMel(0)
        let melMax = hzToMel(sampleRate / 2)

        let melPoints = (0...nMels+1).map {
            melMin + (melMax - melMin) * Float($0) / Float(nMels + 1)
        }

        let hzPoints = melPoints.map(melToHz)
        let bins = hzPoints.map {
            Int(floor((Float(nFFT) + 1) * $0 / sampleRate))
        }

        for m in 1...nMels {
            for k in bins[m-1]..<bins[m] {
                filterbank[m-1][k] =
                    Float(k - bins[m-1]) / Float(bins[m] - bins[m-1])
            }
            for k in bins[m]..<bins[m+1] {
                filterbank[m-1][k] =
                    Float(bins[m+1] - k) / Float(bins[m+1] - bins[m])
            }
        }

        return filterbank
    }
}
