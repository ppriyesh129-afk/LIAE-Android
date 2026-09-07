package com.liae.android

import android.graphics.Bitmap

class FaceSwapPipeline(
    private val detector: BlazeFaceDetector,
    private val liae: LiaeUdEngine
) {

    data class Result(
        val bitmap: Bitmap,
        val facesDetected: Int,
        val debugText: String
    )

    private var lastDebugText = ""

    fun swap(
        sourceImage: Bitmap,
        targetImage: Bitmap
    ): Result {

        val sourceFaces = detector.detect(sourceImage)
        val targetFaces = detector.detect(targetImage)

        if (sourceFaces.isEmpty()) {
            throw IllegalStateException("No face detected in source image")
        }

        if (targetFaces.isEmpty()) {
            throw IllegalStateException("No face detected in target image")
        }

        val sourceFace =
            sourceFaces.maxByOrNull { it.score }
                ?: throw IllegalStateException("Source face selection failed")

        val debugBitmap = swapOneFace(
            sourceImage,
            sourceFace,
            targetImage,
            targetFaces.first()
        )

        return Result(
            bitmap = debugBitmap,
            facesDetected = targetFaces.size,
            debugText = lastDebugText
        )
    }

    private fun swapOneFace(
        sourceImage: Bitmap,
        sourceFace: BlazeFaceResult,
        targetImage: Bitmap,
        targetFace: BlazeFaceResult
    ): Bitmap {

        val alignedSource = DflAligner.align(sourceImage, sourceFace)
        val alignedTarget = DflAligner.align(targetImage, targetFace)

        try {

            val targetTensor =
                ImageTensor.bitmapToTensor(alignedTarget.bitmap)

            val prediction =
                liae.run(targetTensor)

            lastDebugText =
                prediction.debugText

            /*
             * DEBUG:
             * Show the exact aligned 128×128 crop that enters LIAE.
             *
             * No warp.
             * No mask.
             * No merge.
             */
            return alignedTarget.bitmap.copy(
                Bitmap.Config.ARGB_8888,
                false
            )

        } finally {

            if (!alignedSource.bitmap.isRecycled) {
                alignedSource.bitmap.recycle()
            }

            if (!alignedTarget.bitmap.isRecycled) {
                alignedTarget.bitmap.recycle()
            }
        }
    }

    fun close() {
        liae.close()
    }
}
