package com.liae.android

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint

class FaceSwapPipeline(
    private val detector: BlazeFaceDetector,
    private val liae: LiaeUdEngine
) {

    data class Result(
        val bitmap: Bitmap,
        val facesDetected: Int,
        val debugText: String
    )

    private var lastDebugText =
        ""

    fun swap(
        sourceImage: Bitmap,
        targetImage: Bitmap
    ): Result {

        val sourceFaces =
            detector.detect(sourceImage)

        val targetFaces =
            detector.detect(targetImage)

        if (sourceFaces.isEmpty()) {
            throw IllegalStateException(
                "No face detected in source image"
            )
        }

        if (targetFaces.isEmpty()) {
            throw IllegalStateException(
                "No face detected in target image"
            )
        }

        val sourceFace =
            sourceFaces.maxByOrNull {
                it.score
            }
                ?: throw IllegalStateException(
                    "Source face selection failed"
                )

        var result =
            targetImage.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        for (targetFace in targetFaces) {

            val next =
                swapOneFace(
                    sourceImage,
                    sourceFace,
                    result,
                    targetFace
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

        val alignedSource =
            DflAligner.align(
                sourceImage,
                sourceFace
            )

        val alignedTarget =
            DflAligner.align(
                targetImage,
                targetFace
            )

        try {

            val sourceTensor =
                ImageTensor.bitmapToTensor(
                    alignedSource.bitmap
                )

            val targetTensor =
                ImageTensor.bitmapToTensor(
                    alignedTarget.bitmap
                )

            val prediction =
                liae.run(
                    src = sourceTensor,
                    dst = targetTensor
                )

            lastDebugText =
                prediction.debugText

            val swappedFace =
                ImageTensor.tensorToBitmap(
                    prediction.rgb
                )

            val mask =
                ImageTensor.maskToBitmap(
                    prediction.mask
                )

            try {

                val warpedFace =
                    warpToTarget(
                        swappedFace,
                        alignedTarget.inverse,
                        targetImage.width,
                        targetImage.height
                    )

                val warpedMask =
                    warpToTarget(
                        mask,
                        alignedTarget.inverse,
                        targetImage.width,
                        targetImage.height
                    )

                try {

                    return DflMerger.merge(
                        background = targetImage,
                        warpedFace = warpedFace,
                        warpedMask = warpedMask
                    )

                } finally {

                    warpedFace.recycle()
                    warpedMask.recycle()
                }

            } finally {

                swappedFace.recycle()
                mask.recycle()
            }

        } finally {

            alignedSource.bitmap.recycle()
            alignedTarget.bitmap.recycle()
        }
    }

    private fun warpToTarget(
        alignedFace: Bitmap,
        inverse: FloatArray,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {

        val matrix =
            Matrix()

        matrix.setValues(
            floatArrayOf(
                inverse[0],
                inverse[1],
                inverse[2],

                inverse[3],
                inverse[4],
                inverse[5],

                0f,
                0f,
                1f
            )
        )

        val output =
            Bitmap.createBitmap(
                targetWidth,
                targetHeight,
                Bitmap.Config.ARGB_8888
            )

        Canvas(output).drawBitmap(
            alignedFace,
            matrix,
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                    Paint.FILTER_BITMAP_FLAG
            )
        )

        return output
    }

    fun close() {
        liae.close()
    }
}
