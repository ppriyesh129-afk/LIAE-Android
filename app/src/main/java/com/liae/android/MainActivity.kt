package com.liae.android

import ai.onnxruntime.OrtEnvironment
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView
    private lateinit var resultImage: ImageView
    private lateinit var sourceButton: Button
    private lateinit var targetButton: Button

    private var sourceBitmap: Bitmap? = null
    private var targetBitmap: Bitmap? = null

    private var pipeline: FaceSwapPipeline? = null

    private val executor =
        Executors.newSingleThreadExecutor()

    private val sourcePicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri: Uri? ->

            if (uri == null) return@registerForActivityResult

            executor.execute {

                try {

                    val bitmap =
                        loadBitmap(uri)
                            ?: throw IllegalStateException(
                                "Could not load source image"
                            )

                    sourceBitmap?.recycle()
                    sourceBitmap = bitmap

                    runOnUiThread {

                        statusText.text =
                            "Source selected. Select target image."

                        targetButton.isEnabled = true
                    }

                } catch (e: Throwable) {

                    runOnUiThread {

                        statusText.text =
                            "SOURCE ERROR: " +
                                (e.message
                                    ?: e.javaClass.simpleName)
                    }
                }
            }
        }

    private val targetPicker =
        registerForActivityResult(
            ActivityResultContracts.GetContent()
        ) { uri: Uri? ->

            if (uri == null) return@registerForActivityResult

            executor.execute {

                try {

                    val bitmap =
                        loadBitmap(uri)
                            ?: throw IllegalStateException(
                                "Could not load target image"
                            )

                    targetBitmap?.recycle()
                    targetBitmap = bitmap

                    runOnUiThread {

                        statusText.text =
                            "Target selected. Starting face swap..."

                        sourceButton.isEnabled = false
                        targetButton.isEnabled = false
                    }

                    runFaceSwap()

                } catch (e: Throwable) {

                    e.printStackTrace()

                    runOnUiThread {

                        statusText.text =
                            "TARGET ERROR: " +
                                (e.message
                                    ?: e.javaClass.simpleName)

                        sourceButton.isEnabled = true
                        targetButton.isEnabled = true
                    }
                }
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

        sourceButton =
            findViewById(R.id.sourceButton)

        targetButton =
            findViewById(R.id.targetButton)

        targetButton.isEnabled = false

        statusText.text = "Select source face"

        sourceButton.setOnClickListener {

            sourcePicker.launch("image/*")
        }

        targetButton.setOnClickListener {

            if (sourceBitmap == null) {

                statusText.text =
                    "Select source image first"

                return@setOnClickListener
            }

            targetPicker.launch("image/*")
        }
    }

    private fun loadBitmap(
        uri: Uri
    ): Bitmap? {

        return contentResolver
            .openInputStream(uri)
            .use { input ->

                if (input == null) {
                    return null
                }

                BitmapFactory.decodeStream(input)
            }
    }

    private fun runFaceSwap() {

        val source =
            sourceBitmap
                ?: throw IllegalStateException(
                    "Source image missing"
                )

        val target =
            targetBitmap
                ?: throw IllegalStateException(
                    "Target image missing"
                )

        try {

            runOnUiThread {

                statusText.text =
                    "Loading BlazeFace + LIAE..."
            }

            if (pipeline == null) {

                val modelFile =
                    getFileStreamPath("blazeface.onnx")

                if (!modelFile.exists()) {

                    assets.open("blazeface.onnx").use { input ->

                        modelFile.outputStream().use { output ->

                            input.copyTo(output)
                        }
                    }
                }

                val env =
                    OrtEnvironment.getEnvironment()

                val blazeSession =
                    env.createSession(
                        modelFile.absolutePath,
                        ai.onnxruntime.OrtSession.SessionOptions()
                    )

                val detector =
                    BlazeFaceDetector(blazeSession)

                val liae =
                    LiaeUdEngine(this)

                pipeline =
                    FaceSwapPipeline(
                        detector,
                        liae
                    )
            }

            runOnUiThread {

                statusText.text =
                    "Detecting faces..."
            }

            val currentPipeline =
                pipeline
                    ?: throw IllegalStateException(
                        "Pipeline initialization failed"
                    )

            runOnUiThread {

                statusText.text =
                    "Running LIAE face swap..."
            }

            val result =
                currentPipeline.swap(
                    sourceImage = source,
                    targetImage = target
                )

            runOnUiThread {

                resultImage.setImageBitmap(
                    result.bitmap
                )

                statusText.text =
                    "Face swap complete\n" +
                        "Faces detected: ${result.facesDetected}"

                sourceButton.isEnabled = true
                targetButton.isEnabled = true
            }

        } catch (e: Throwable) {

            e.printStackTrace()

            runOnUiThread {

                statusText.text =
                    "FACE SWAP ERROR\n\n" +
                        e.javaClass.simpleName +
                        "\n" +
                        (e.message ?: "Unknown error")

                sourceButton.isEnabled = true
                targetButton.isEnabled = true
            }
        }
    }

    override fun onDestroy() {

        executor.shutdownNow()

        pipeline?.close()
        pipeline = null

        sourceBitmap?.recycle()
        targetBitmap?.recycle()

        sourceBitmap = null
        targetBitmap = null

        super.onDestroy()
    }
}
