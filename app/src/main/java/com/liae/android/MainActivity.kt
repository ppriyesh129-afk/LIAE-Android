package com.liae.android

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var engine: LiaeUdEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        val status = findViewById<TextView>(R.id.statusText)

        try {
            engine = LiaeUdEngine(this)

            status.text = "LIAE ONNX model loaded successfully"

        } catch (e: Exception) {
            status.text = "Model load failed:\n${e.message}"
        }
    }

    override fun onDestroy() {
        if (::engine.isInitialized) {
            engine.close()
        }

        super.onDestroy()
    }
}
