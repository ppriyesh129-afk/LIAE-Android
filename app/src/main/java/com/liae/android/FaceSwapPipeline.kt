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
        val facesDetected: Int
    )

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

        var result =
            targetImage.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        for (targetFace in targetFaces) {

            result =
                swapOneFace(
                    sourceImage,
                    sourceFace,
                    result,
                    targetFace
                )
        }

        return Result(
            bitmap = result,
            facesDetected = targetFaces.size
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

            val swappedFace =
                ImageTensor.tensorToBitmap(
                    prediction.rgb
                )

            val mask =
                createMaskBitmap(
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

                val output =
                    blend(
                        targetImage,
                        warpedFace,
                        warpedMask
                    )

                warpedFace.recycle()
                warpedMask.recycle()

                return output

            } finally {

                swappedFace.recycle()
                mask.recycle()
            }

        } finally {

            alignedSource.bitmap.recycle()
            alignedTarget.bitmap.recycle()
        }
    }

    private fun createMaskBitmap(
        mask: FloatArray
    ): Bitmap {

        require(mask.size == 128 * 128)

        val pixels =
            IntArray(128 * 128)

        for (i in pixels.indices) {

            val alpha =
                (mask[i]
                    .coerceIn(0f, 1f) * 255f)
                    .toInt()

            pixels[i] =
                (alpha shl 24) or
                        0x00FFFFFF
        }

        return Bitmap.createBitmap(
            pixels,
            128,
            128,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun warpToTarget(
        alignedFace: Bitmap,
        inverse: FloatArray,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {

        val matrix = Matrix()

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

    private fun blend(
        background: Bitmap,
        foreground: Bitmap,
        mask: Bitmap
    ): Bitmap {

        val width = background.width
        val height = background.height

        val bg = IntArray(width * height)
        val fg = IntArray(width * height)
        val mk = IntArray(width * height)
        val out = IntArray(width * height)

        background.getPixels(bg, 0, width, 0, 0, width, height)
        foreground.getPixels(fg, 0, width, 0, 0, width, height)
        mask.getPixels(mk, 0, width, 0, 0, width, height)

        for (i in out.indices) {

            val alpha =
                smoothStep(
                    0.05f,
                    0.95f,
                    (mk[i] ushr 24) / 255f
                )

            val br = (bg[i] shr 16) and 255
            val bgc = (bg[i] shr 8) and 255
            val bb = bg[i] and 255

            val fr = (fg[i] shr 16) and 255
            val fgc = (fg[i] shr 8) and 255
            val fb = fg[i] and 255

            val r =
                (br * (1 - alpha) + fr * alpha)
                    .toInt()
                    .coerceIn(0, 255)

            val g =
                (bgc * (1 - alpha) + fgc * alpha)
                    .toInt()
                    .coerceIn(0, 255)

            val b =
                (bb * (1 - alpha) + fb * alpha)
                    .toInt()
                    .coerceIn(0, 255)

            out[i] =
                (0xFF shl 24) or
                        (r shl 16) or
                        (g shl 8) or
                        b
        }

        return Bitmap.createBitmap(
            out,
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
    }

    private fun smoothStep(
        edge0: Float,
        edge1: Float,
        value: Float
    ): Float {

        val x =
            ((value - edge0) /
                    (edge1 - edge0))
                .coerceIn(0f, 1f)

        return x * x * (3f - 2f * x)
    }

    fun close() {
        liae.close()
    }
}
