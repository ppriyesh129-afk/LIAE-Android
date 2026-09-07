package com.liae.android

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var engine: LiaeUdEngine

    private lateinit var statusText: TextView
    private lateinit var resultImage: ImageView

    private val imagePicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri: Uri? ->

            if (uri != null) {
                processImage(uri)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        resultImage = findViewById(R.id.resultImage)

        engine = LiaeUdEngine(this)

        findViewById<Button>(R.id.selectButton).setOnClickListener {
            imagePicker.launch("image/*")
        }
    }

    private fun processImage(uri: Uri) {

        try {

            statusText.text = "Loading image..."

            val bitmap = MediaStore.Images.Media.getBitmap(
                contentResolver,
                uri
            )

            statusText.text = "Preparing tensor..."

            val tensor = ImageTensor.bitmapToTensor(bitmap)

            statusText.text = "Running LIAE..."

            val result = engine.run(
                src = tensor,
                dst = tensor
            )

            val outputBitmap =
                ImageTensor.tensorToBitmap(result.rgb)

            resultImage.setImageBitmap(outputBitmap)

            statusText.text =
                """
                LIAE inference complete

                RGB: ${result.rgb.size}
                Mask: ${result.mask.size}
                """.trimIndent()

        } catch (e: Exception) {

            statusText.text =
                """
                Error

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
