package com.liae.android

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import java.io.InputStream
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    companion object {
        private const val REQUEST_SOURCE = 1001
        private const val REQUEST_TARGET = 1002
    }

    private lateinit var selectButton: Button
    private lateinit var resultImage: ImageView
    private lateinit var statusText: TextView

    private var sourceBitmap: Bitmap? = null
    private var targetBitmap: Bitmap? = null

    private var pipeline: FaceSwapPipeline? = null

    private val executor =
        Executors.newSingleThreadExecutor()

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        setContentView(
            R.layout.activity_main
        )

        selectButton =
            findViewById(
                R.id.selectButton
            )

        resultImage =
            findViewById(
                R.id.resultImage
            )

        statusText =
            findViewById(
                R.id.statusText
            )

        selectButton.setOnClickListener {

            chooseSourceImage()
        }

        statusText.text =
            "Select source face"
    }

    private fun chooseSourceImage() {

        val intent =
            Intent(
                Intent.ACTION_OPEN_DOCUMENT
            )

        intent.type =
            "image/*"

        intent.addCategory(
            Intent.CATEGORY_OPENABLE
        )

        startActivityForResult(
            intent,
            REQUEST_SOURCE
        )
    }

    private fun chooseTargetImage() {

        val intent =
            Intent(
                Intent.ACTION_OPEN_DOCUMENT
            )

        intent.type =
            "image/*"

        intent.addCategory(
            Intent.CATEGORY_OPENABLE
        )

        startActivityForResult(
            intent,
            REQUEST_TARGET
        )
    }

    @Deprecated(
        "Use Activity Result API in future"
    )
    override fun onActivityResult(
        requestCode: Int,
        resultCode: Int,
        data: Intent?
    ) {
        super.onActivityResult(
            requestCode,
            resultCode,
            data
        )

        if (
            resultCode != Activity.RESULT_OK ||
            data?.data == null
        ) {
            return
        }

        val uri =
            data.data!!

        when (requestCode) {

            REQUEST_SOURCE -> {

                val bitmap =
                    loadBitmap(uri)

                if (bitmap == null) {

                    statusText.text =
                        "Could not load source image"

                    return
                }

                sourceBitmap?.recycle()

                sourceBitmap =
                    bitmap

                statusText.text =
                    "Source selected. Select target face."

                selectButton.text =
                    "Select Target Image"

                selectButton.setOnClickListener {

                    chooseTargetImage()
                }
            }

            REQUEST_TARGET -> {

                val bitmap =
                    loadBitmap(uri)

                if (bitmap == null) {

                    statusText.text =
                        "Could not load target image"

                    return
                }

                targetBitmap?.recycle()

                targetBitmap =
                    bitmap

                statusText.text =
                    "Target selected. Starting face swap..."

                selectButton.isEnabled =
                    false

                runFaceSwap()
            }
        }
    }

    private fun loadBitmap(
        uri: Uri
    ): Bitmap? {

        return try {

            contentResolver.openInputStream(
                uri
            ).use { input ->

                if (input == null) {
                    return null
                }

                BitmapFactory.decodeStream(
                    input
                )
            }

        } catch (
            e: Exception
        ) {

            e.printStackTrace()

            null
        }
    }

    private fun runFaceSwap() {

        val source =
            sourceBitmap

        val target =
            targetBitmap

        if (
            source == null ||
            target == null
        ) {

            statusText.text =
                "Source or target image missing"

            selectButton.isEnabled =
                true

            return
        }

        executor.execute {

            try {

                /*
                 * Create the heavy ONNX pipeline
                 * only when the swap is requested.
                 */
                if (pipeline == null) {

                    runOnUiThread {

                        statusText.text =
                            "Loading LIAE-UD model..."
                    }

                    val detector =
                        YuNetDetector(
                            this@MainActivity
                        )

                    val liae =
                        LiaeUdEngine(
                            this@MainActivity
                        )

                    pipeline =
                        FaceSwapPipeline(
                            detector,
                            liae
                        )
                }

                runOnUiThread {

                    statusText.text =
                        "Running face detection..."
                }

                val currentPipeline =
                    pipeline
                        ?: throw IllegalStateException(
                            "Pipeline initialization failed"
                        )

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
                        "Face swap complete. " +
                            "Faces: " +
                            result.facesDetected

                    selectButton.isEnabled =
                        true

                    selectButton.text =
                        "Select New Source"

                    selectButton.setOnClickListener {

                        chooseSourceImage()
                    }
                }

            } catch (
                e: Exception
            ) {

                e.printStackTrace()

                val message =
                    e.message
                        ?: e.javaClass.simpleName

                runOnUiThread {

                    statusText.text =
                        "ERROR: $message"

                    selectButton.isEnabled =
                        true
                }
            }
        }
    }

    override fun onDestroy() {

        super.onDestroy()

        executor.shutdownNow()

        pipeline?.close()

        pipeline = null

        sourceBitmap?.recycle()
        targetBitmap?.recycle()

        sourceBitmap = null
        targetBitmap = null
    }
}
