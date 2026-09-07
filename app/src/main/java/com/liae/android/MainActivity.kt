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

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        setContentView(
            R.layout.activity_main
        )

        statusText =
            findViewById(R.id.statusText)

        resultImage =
            findViewById(R.id.resultImage)

        selectButton =
            findViewById(R.id.selectButton)

        statusText.text = "Ready"

        selectButton.setOnClickListener {

            imagePicker.launch(
                "image/*"
            )
        }
    }

    private fun processImage(
        uri: Uri
    ) {

        selectButton.isEnabled = false

        statusText.text =
            "Preparing..."

        thread {

            var bitmap: Bitmap? = null

            try {

                runOnUiThread {

                    statusText.text =
                        "Loading LIAE model..."
                }

                /*
                 * Load the large ONNX model only when
                 * the user actually selects an image.
                 */
                if (engine == null) {

                    engine =
                        LiaeUdEngine(this)
                }

                runOnUiThread {

                    statusText.text =
                        "Loading image..."
                }

                bitmap =
                    MediaStore.Images.Media.getBitmap(
                        contentResolver,
                        uri
                    )

                runOnUiThread {

                    statusText.text =
                        "Preparing tensor..."
                }

                /*
                 * Convert image to:
                 *
                 * 128 x 128 x 3
                 * RGB
                 * 0..1
                 */
                val tensor =
                    ImageTensor.bitmapToTensor(
                        bitmap
                    )

                runOnUiThread {

                    statusText.text =
                        "Running LIAE..."
                }

                /*
                 * Current stage is an inference test.
                 *
                 * Source and destination are temporarily
                 * the same image.
                 */
                val result =
                    engine!!.run(
                        src = tensor,
                        dst = tensor
                    )

                /*
                 * Apply the LIAE mask to the RGB output.
                 *
                 * This prevents the generated output from
                 * being displayed outside the predicted face
                 * region.
                 */
                val maskedRgb =
                    ImageTensor.applyMask(
                        rgb = result.rgb,
                        mask = result.mask,
                        background = tensor
                    )

                /*
                 * Convert model output into an Android Bitmap.
                 *
                 * The converter handles:
                 *
                 * 0..1
                 * -1..1
                 * NaN
                 * Infinity
                 */
                val outputBitmap =
                    ImageTensor.modelRgbToBitmap(
                        tensor = maskedRgb,
                        bgr = false
                    )

                runOnUiThread {

                    resultImage.setImageBitmap(
                        outputBitmap
                    )

                    statusText.text =
                        """
                        LIAE inference complete

                        RGB: ${result.rgb.size}
                        Mask: ${result.mask.size}

                        RGB min: ${result.rgbMin}
                        RGB max: ${result.rgbMax}
                        RGB mean: ${result.rgbMean}

                        Mask min: ${result.maskMin}
                        Mask max: ${result.maskMax}
                        Mask mean: ${result.maskMean}
                        """.trimIndent()

                    selectButton.isEnabled =
                        true
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

                    statusText.text =
                        error

                    selectButton.isEnabled =
                        true
                }

            } finally {

                /*
                 * Release the full-resolution source bitmap.
                 *
                 * The 128x128 output bitmap remains displayed.
                 */
                bitmap?.recycle()
            }
        }
    }

    override fun onDestroy() {

        engine?.close()

        engine = null

        super.onDestroy()
    }
}

Now you have 3 updated pieces

1. "LiaeUdEngine.kt"
Diagnostic RGB/mask statistics.

2. "ImageTensor.kt"
RGB normalization + channel handling + mask application.

3. "MainActivity.kt"
Uses the new processing pipeline.

Build a new APK and test one image.

Then send me the displayed:

RGB min:
RGB max:
RGB mean:

Mask min:
Mask max:
Mask mean:

Those values are important before we make the next LIAE correction.
