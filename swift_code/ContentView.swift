//
//  ContentView.swift
//  SnoreDetector
//
//  Created by Kenny Withers on 12/18/25.
//

import SwiftUI


/*struct ContentView: View {
    let audio = AudioManager.shared

    var body: some View {
        Text("Snore Detector Running")
            .onAppear {
                print("📱 ContentView appeared")
            }
    }
}

#Preview {
    ContentView()
}*/

struct ContentView: View {
    // 1. Your variable goes here (inside the struct, but outside 'body')
    let audio = AudioManager.shared

    var body: some View {
        // 2. The visual layout goes inside 'body'
        VStack {
            Image(systemName: "globe")
                .imageScale(.large)
                .foregroundStyle(.tint)
            
            Text("Hello, world!")
        }
        .padding()
        // 3. The logic to run when the screen loads goes here
        .onAppear {
            print("📱 ContentView appeared")
            
            // This confirms your audio variable is accessible
            print("Audio Manager is ready: \(audio)")
            AudioManager.shared.startScanning()
        }
    }
}
