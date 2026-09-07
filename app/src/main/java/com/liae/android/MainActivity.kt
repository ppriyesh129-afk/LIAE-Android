package com.liae.android

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private var engine: LiaeUdEngine? = null

    private lateinit var statusText: TextView
    private lateinit var resultImage: ImageView
    private lateinit var selectButton: Button

    private val imagePicker =
        registerForActivityResult(
            androidx.activity.result.contract.ActivityResultContracts.GetContent()
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
        selectButton = findViewById(R.id.selectButton)

        statusText.text = "Ready"

        selectButton.setOnClickListener {
            imagePicker.launch("image/*")
        }
    }

    private fun processImage(uri: Uri) {

        selectButton.isEnabled = false
        statusText.text = "Preparing..."

        thread {

            try {

                runOnUiThread {
                    statusText.text = "Loading LIAE model..."
                }

                if (engine == null) {
                    engine = LiaeUdEngine(this)
                }

                runOnUiThread {
                    statusText.text = "Loading image..."
                }

                val bitmap: Bitmap =
                    MediaStore.Images.Media.getBitmap(
                        contentResolver,
                        uri
                    )

                runOnUiThread {
                    statusText.text = "Preparing tensor..."
                }

                val tensor =
                    ImageTensor.bitmapToTensor(bitmap)

                runOnUiThread {
                    statusText.text = "Running LIAE..."
                }

                val result =
                    engine!!.run(
                        src = tensor,
                        dst = tensor
                    )

                val outputBitmap =
                    ImageTensor.tensorToBitmap(result.rgb)

                runOnUiThread {

                    resultImage.setImageBitmap(outputBitmap)

                    statusText.text =
                        """
                        LIAE inference complete

                        RGB: ${result.rgb.size}
                        Mask: ${result.mask.size}
                        """.trimIndent()

                    selectButton.isEnabled = true
                }

            } catch (e: Throwable) {

                val error =
                    """
                    LIAE ERROR

                    ${e.javaClass.name}

                    ${e.message}

                    ${e.stackTraceToString()}
                    """.trimIndent()

                runOnUiThread {

                    statusText.text = error
                    selectButton.isEnabled = true
                }
            }
        }
    }

    override fun onDestroy() {

        engine?.close()
        engine = null

        super.onDestroy()
    }
}
