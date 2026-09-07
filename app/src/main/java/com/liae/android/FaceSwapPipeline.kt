package com.liae.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.PorterDuff

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

        // Keep source validation for app flow compatibility.
        val sourceFace = sourceFaces.maxByOrNull { it.score }
            ?: throw IllegalStateException("Source face selection failed")

        @Suppress("UNUSED_VARIABLE")
        val verifiedSourceFace = sourceFace

        var result = targetImage.copy(Bitmap.Config.ARGB_8888, true)

        for (targetFace in targetFaces) {

            val next = swapOneFace(
                sourceImage = sourceImage,
                sourceFace = sourceFace,
                targetImage = result,
                targetFace = targetFace
            )

            if (next !== result) {
                result.recycle()
            }

            result = next
        }

        return Result(
            bitmap = result,
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

            val targetTensor = ImageTensor.bitmapToTensor(alignedTarget.bitmap)

            val prediction = liae.run(targetTensor)

            lastDebugText = prediction.debugText

            val swappedFace = ImageTensor.tensorToBitmap(prediction.rgb)

            // IMPORTANT:
            // Use destination mask for blending back into the target image.
            val mask = ImageTensor.maskToBitmap(prediction.dstMask)

            try {

                val warpedFace = warpToTarget(
                    alignedFace = swappedFace,
                    inverse = alignedTarget.inverse,
                    targetWidth = targetImage.width,
                    targetHeight = targetImage.height
                )

                val warpedMask = warpToTarget(
                    alignedFace = mask,
                    inverse = alignedTarget.inverse,
                    targetWidth = targetImage.width,
                    targetHeight = targetImage.height
                )

                try {

                    return DflMerger.merge(
                        background = targetImage,
                        warpedFace = warpedFace,
                        warpedMask = warpedMask
                    )

                } finally {

                    if (!warpedFace.isRecycled) warpedFace.recycle()
                    if (!warpedMask.isRecycled) warpedMask.recycle()
                }

            } finally {

                if (!swappedFace.isRecycled) swappedFace.recycle()
                if (!mask.isRecycled) mask.recycle()
            }

        } finally {

            if (!alignedSource.bitmap.isRecycled) alignedSource.bitmap.recycle()
            if (!alignedTarget.bitmap.isRecycled) alignedTarget.bitmap.recycle()
        }
    }

    private fun warpToTarget(
        alignedFace: Bitmap,
        inverse: FloatArray,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {

        require(inverse.size == 6) {
            "Expected 6-value affine inverse matrix, got ${inverse.size}"
        }

        val matrix = Matrix()

        matrix.setValues(
            floatArrayOf(
                inverse[0], inverse[1], inverse[2],
                inverse[3], inverse[4], inverse[5],
                0f, 0f, 1f
            )
        )

        val output = Bitmap.createBitmap(
            targetWidth,
            targetHeight,
            Bitmap.Config.ARGB_8888
        )

        val canvas = Canvas(output)

        // Start fully transparent.
        canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.CLEAR)

        val paint = Paint(
            Paint.ANTI_ALIAS_FLAG or Paint.FILTER_BITMAP_FLAG
        )

        paint.isFilterBitmap = true

        canvas.drawBitmap(alignedFace, matrix, paint)

        return output
    }

    fun close() {
        liae.close()
    }
}
