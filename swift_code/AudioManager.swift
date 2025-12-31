//
//  AudioManager.swift
//  SnoreDetector
//
//  Created by Kenny Withers on 12/19/25.
//


//import AVFoundation

/*final class AudioManager {

    static let shared = AudioManager()

    private let snoreModel = SnoreModel()

    private init() {
        print("🎙 AudioManager init")
    }

    func processAudioBuffer(buffer: AVAudioPCMBuffer) {
        let channel = buffer.floatChannelData![0]
        let audio = Array(
            UnsafeBufferPointer(
                start: channel,
                count: Int(buffer.frameLength)
            )
        )

        let mel = MelSpectrogram.compute(audio: audio)
        let prob = snoreModel.predict(mel: mel)

        if prob > 0.5 {
            print("😴 Snoring detected!", prob)
        }
    }
}*/
//import SwiftUI
//import Combine

/*final class AudioManager: ObservableObject {

    static let shared = AudioManager()

    // NEW: The engine that manages the microphone
    private let audioEngine = AVAudioEngine()
    
    // Your existing model
    private let snoreModel = SnoreModel()

    private init() {
        print("🎙 AudioManager init")
    }
    
    // NEW: Function to turn on the microphone
    func startScanning() {
            // 1. Setup the Audio Session (Vital for getting a valid format!)
            let session = AVAudioSession.sharedInstance()
            do {
                // Configure for recording. .measurement mode is good for analysis (no gain correction)
                try session.setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
                // Activate the session - this "wakes up" the hardware
                try session.setActive(true, options: .notifyOthersOnDeactivation)
            } catch {
                print("❌ Failed to set up audio session: \(error)")
                return // Stop if we can't activate the session
            }
            
            // 2. Now it is safe to access inputNode
            let inputNode = audioEngine.inputNode
            let recordingFormat = inputNode.outputFormat(forBus: 0)
            
            // Safety Check: Ensure format is actually valid before tapping
            if recordingFormat.sampleRate == 0 || recordingFormat.channelCount == 0 {
                print("❌ Error: Invalid Audio Format. SampleRate: \(recordingFormat.sampleRate), Channels: \(recordingFormat.channelCount)")
                return
            }
            
            // 3. Remove existing taps (prevents crashes on restart)
            inputNode.removeTap(onBus: 0)

            // 4. Install the Tap
            inputNode.installTap(onBus: 0, bufferSize: 1024, format: recordingFormat) { [weak self] (buffer, time) in
                self?.processAudioBuffer(buffer: buffer)
            }

            // 5. Start the engine
            do {
                try audioEngine.start()
                print("👂 Audio Engine started - Listening at \(recordingFormat.sampleRate)Hz")
            } catch {
                print("❌ Failed to start audio engine: \(error)")
            }
        }

    func processAudioBuffer(buffer: AVAudioPCMBuffer) {
        // Guard against empty buffers
        guard let channelData = buffer.floatChannelData else { return }
        
        let channel = channelData[0]
        let audio = Array(
            UnsafeBufferPointer(
                start: channel,
                count: Int(buffer.frameLength)
            )
        )

        // This calls your class, which triggers the print statements!
        /*let mel = MelSpectrogram.compute(audio: audio)
        
        // Run prediction (Optional: Check if model is ready)
        let prob = snoreModel.predict(mel: mel)
        if prob > 0.5 { print("😴 Snoring detected!", prob) }*/
        
        // Inside processAudioBuffer...

        // This returns MLMultiArray? (Optional) now
        let mel = MelSpectrogram.compute(audio: audio)

        // Pass it directly. The SnoreModel handles the unwrapping safely.
        let prob = snoreModel.predict(mel: mel)

        if prob > 0.5 {
            print("😴 Snoring detected! Probability: \(prob)")
        }
    }
}*/


import SwiftUI
import Combine
import AVFoundation

final class AudioManager: ObservableObject {

    static let shared = AudioManager()

    private let audioEngine = AVAudioEngine()
    private let snoreModel = SnoreModel()
    
    // 1️⃣ PROCESSING QUEUE (Background Thread)
    private let processingQueue = DispatchQueue(label: "com.snoredetector.processing", qos: .userInitiated)
    
    // 2️⃣ BUFFERS
    private var audioBuffer: [Float] = []
    // We need slightly more than 16k to fill 63 frames comfortably without padding artifacts
    // 63 frames * 256 hop + 1024 window = ~17k. Let's buffer 16k and let padding handle the rest.
    private let requiredSampleCount = 16_000
    
    // 3️⃣ CONVERTER PROPERTIES
    private var converter: AVAudioConverter?
    private let targetSampleRate: Double = 16_000.0 // Model requires 16k
    
    private init() {
        print("🎙 AudioManager init")
    }
    
    func startScanning() {
        let session = AVAudioSession.sharedInstance()
        do {
            // .measurement is good, but let's default to .playAndRecord for broader support
            try session.setCategory(.playAndRecord, mode: .measurement, options: .duckOthers)
            try session.setActive(true, options: .notifyOthersOnDeactivation)
        } catch {
            print("❌ Failed to set up audio session: \(error)")
            return
        }
            
        let inputNode = audioEngine.inputNode
        // This is likely 48,000 Hz or 44,100 Hz
        let nativeFormat = inputNode.outputFormat(forBus: 0)
            
        if nativeFormat.sampleRate == 0 || nativeFormat.channelCount == 0 {
            return
        }
        
        // 4️⃣ SETUP CONVERTER
        // We want 16kHz, 1 Channel (Mono), Float32
        guard let targetFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: targetSampleRate, channels: 1, interleaved: false) else {
            print("❌ Failed to create target audio format")
            return
        }
        
        self.converter = AVAudioConverter(from: nativeFormat, to: targetFormat)
            
        inputNode.removeTap(onBus: 0)

        // 5️⃣ INSTALL TAP
        // We tap the Native Format (48k), but we will convert it inside the block
        inputNode.installTap(onBus: 0, bufferSize: 1024, format: nativeFormat) { [weak self] (buffer, time) in
            self?.handleAudioInput(nativeBuffer: buffer, targetFormat: targetFormat)
        }

        do {
            try audioEngine.start()
            print("👂 Audio Engine started - Native: \(nativeFormat.sampleRate)Hz -> Converting to: 16000Hz")
        } catch {
            print("❌ Failed to start audio engine: \(error)")
        }
    }

    private func handleAudioInput(nativeBuffer: AVAudioPCMBuffer, targetFormat: AVAudioFormat) {
        // Run conversion and processing on background thread
        processingQueue.async { [weak self] in
            guard let self = self, let converter = self.converter else { return }
            
            // 1. Calculate how many frames the 16k buffer needs
            // Ratio: 16000 / 48000 = 0.33. So 1024 input frames -> ~341 output frames
            let inputFrameCount = AVAudioFrameCount(nativeBuffer.frameLength)
            let conversionRatio = targetFormat.sampleRate / nativeBuffer.format.sampleRate
            let outputFrameCount = AVAudioFrameCount(Double(inputFrameCount) * conversionRatio)
            
            // 2. Create output buffer
            guard let convertedBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outputFrameCount) else { return }
            
            // 3. Perform Conversion
            var error: NSError? = nil
            let status = converter.convert(to: convertedBuffer, error: &error) { packetCount, inputStatus in
                // We give the converter our native data here
                inputStatus.pointee = .haveData
                return nativeBuffer
            }
            
            if status == .error || error != nil {
                print("⚠️ Conversion Error")
                return
            }
            
            // 4. Process the clean 16k audio
            self.processConvertedBuffer(convertedBuffer)
        }
    }
    
    private func processConvertedBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let channelData = buffer.floatChannelData else { return }
        
        // Extract 16k Data
        let channel = channelData[0]
        let chunk = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
        
        // 1️⃣ DEBUG: Calculate Volume
        /*var sum: Float = 0
        for x in chunk { sum += x * x }
        let rms = sqrt(sum / Float(chunk.count))
            
        // 2️⃣ PRINT IT: If this is 0.0000, your microphone code is broken
        print("🔊 Input Volume: \(String(format: "%.5f", rms))")*/
        
        // Add to Bucket
        self.audioBuffer.append(contentsOf: chunk)
        
        // Check if bucket is full
        if self.audioBuffer.count >= self.requiredSampleCount {
            
            let analysisChunk = Array(self.audioBuffer.prefix(self.requiredSampleCount))
            
            // Slide window: Remove half
            self.audioBuffer.removeFirst(self.requiredSampleCount / 2)
            
            // Run Detection
            self.runDetection(on: analysisChunk)
        }
    }
    
    private func runDetection(on audio: [Float]) {
        guard let mel = MelSpectrogram.compute(audio: audio) else { return }
        
        let prob = snoreModel.predict(mel: mel)
        
        // Lower threshold slightly for testing (0.3) then move back to 0.5 or 0.8
        if prob > 0.5 {
            print("😴 SNORE DETECTED! Prob: \(String(format: "%.2f", prob))")
            
            DispatchQueue.main.async {
               // Update UI here
            }
        }
        /*else{
            print("Current Detection Prob: \(String(format: "%.2f", prob))")
        }*/
    }
}
