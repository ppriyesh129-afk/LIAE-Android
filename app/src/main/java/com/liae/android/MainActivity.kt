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
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity() {

    private var engine: LiaeUdEngine? = null

    private lateinit var statusText: TextView
    private lateinit var resultImage: ImageView
    private lateinit var sourceButton: Button
    private lateinit var targetButton: Button

    private var sourceBitmap: Bitmap? = null
    private var targetBitmap: Bitmap? = null

    private val sourcePicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                sourceBitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
                statusText.text = "Source selected"
                startIfReady()
            }
        }

    private val targetPicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                targetBitmap = MediaStore.Images.Media.getBitmap(contentResolver, it)
                statusText.text = "Target selected"
                startIfReady()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)
        resultImage = findViewById(R.id.resultImage)
        sourceButton = findViewById(R.id.sourceButton)
        targetButton = findViewById(R.id.targetButton)

        statusText.text = "Select Source Face"

        sourceButton.setOnClickListener {
            sourcePicker.launch("image/*")
        }

        targetButton.setOnClickListener {
            targetPicker.launch("image/*")
        }
    }

    private fun startIfReady() {
        val src = sourceBitmap ?: return
        val dst = targetBitmap ?: return
        runSwap(src, dst)
    }

    private fun runSwap(srcBitmap: Bitmap, dstBitmap: Bitmap) {

        sourceButton.isEnabled = false
        targetButton.isEnabled = false

        statusText.text = "Preparing..."

        thread {
            try {

                if (engine == null) {
                    runOnUiThread {
                        statusText.text = "Loading LIAE model..."
                    }
                    engine = LiaeUdEngine(this)
                }

                runOnUiThread {
                    statusText.text = "Preparing tensors..."
                }

                val srcTensor = ImageTensor.bitmapToTensor(srcBitmap)
                val dstTensor = ImageTensor.bitmapToTensor(dstBitmap)

                runOnUiThread {
                    statusText.text = "Running LIAE..."
                }

                val result = engine!!.run(
                    src = srcTensor,
                    dst = dstTensor
                )

                val rgbBitmap = ImageTensor.tensorToBitmap(result.rgb)
                val maskBitmap = ImageTensor.maskToBitmap(result.mask)

                runOnUiThread {

                    // Show RGB output (the swapped face)
                    resultImage.setImageBitmap(rgbBitmap)

                    statusText.text = """
                        LIAE inference complete

                        RGB: ${result.rgb.size}
                        Mask: ${result.mask.size}

                        RGB min=${result.rgb.minOrNull()}
                        RGB max=${result.rgb.maxOrNull()}

                        Mask min=${result.mask.minOrNull()}
                        Mask max=${result.mask.maxOrNull()}
                    """.trimIndent()

                    sourceButton.isEnabled = true
                    targetButton.isEnabled = true
                }

            } catch (e: Throwable) {

                runOnUiThread {

                    statusText.text = """
                        LIAE ERROR

                        ${e.javaClass.simpleName}

                        ${e.message}
                    """.trimIndent()

                    sourceButton.isEnabled = true
                    targetButton.isEnabled = true
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
