package com.liae.android

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.random.Random

class MainActivity : AppCompatActivity() {

    private lateinit var engine: LiaeUdEngine

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        val status = findViewById<TextView>(R.id.statusText)

        try {
            status.text = "Loading LIAE model..."

            engine = LiaeUdEngine(this)

            val src = FloatArray(128 * 128 * 3) {
                Random.nextFloat()
            }

            val dst = FloatArray(128 * 128 * 3) {
                Random.nextFloat()
            }

            status.text = "Running LIAE inference..."

            val result = engine.run(src, dst)

            status.text =
                """
                LIAE inference SUCCESS

                RGB floats: ${result.rgb.size}
                Mask floats: ${result.mask.size}

                Expected:
                RGB  = 49152
                Mask = 16384
                """.trimIndent()

        } catch (e: Exception) {

            status.text =
                """
                LIAE inference FAILED

                ${e.javaClass.simpleName}

                ${e.message}
                """.trimIndent()
        }
    }

    override fun onDestroy() {
        if (::engine.isInitialized) {
            engine.close()
        }

        super.onDestroy()
    }
}
