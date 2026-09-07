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
            sourceFaces.maxByOrNull { it.score }
                ?: throw IllegalStateException(
                    "Source face selection failed"
                )

        var result =
            targetImage.copy(
                Bitmap.Config.ARGB_8888,
                true
            )

        try {

            for (targetFace in targetFaces) {

                val previous = result

                result =
                    swapOneFace(
                        sourceImage = sourceImage,
                        sourceFace = sourceFace,
                        targetImage = previous,
                        targetFace = targetFace
                    )

                if (result !== previous && !previous.isRecycled) {
                    previous.recycle()
                }
            }

            return Result(
                bitmap = result,
                facesDetected = targetFaces.size
            )

        } catch (e: Throwable) {

            if (!result.isRecycled) {
                result.recycle()
            }

            throw e
        }
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

            /*
             * IMPORTANT:
             *
             * DflAligner produces the 128x128 aligned face.
             * ImageTensor must receive that image directly.
             *
             * No detector box is used here.
             */

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

            require(
                prediction.rgb.size == 128 * 128 * 3
            ) {
                "Invalid LIAE RGB output: ${prediction.rgb.size}"
            }

            require(
                prediction.mask.size == 128 * 128
            ) {
                "Invalid LIAE mask output: ${prediction.mask.size}"
            }

            val swappedFace =
                ImageTensor.tensorToBitmap(
                    prediction.rgb
                )

            val mask =
                ImageTensor.maskToBitmap(
                    prediction.mask
                )

            try {

                /*
                 * Transform the generated 128x128 face
                 * back into the ORIGINAL target image.
                 */
                val warpedFace =
                    warpToTarget(
                        alignedFace = swappedFace,
                        inverse = alignedTarget.inverse,
                        targetWidth = targetImage.width,
                        targetHeight = targetImage.height
                    )

                /*
                 * Transform the LIAE mask using exactly
                 * the same transform.
                 */
                val warpedMask =
                    warpToTarget(
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

                    if (!warpedFace.isRecycled) {
                        warpedFace.recycle()
                    }

                    if (!warpedMask.isRecycled) {
                        warpedMask.recycle()
                    }
                }

            } finally {

                if (!swappedFace.isRecycled) {
                    swappedFace.recycle()
                }

                if (!mask.isRecycled) {
                    mask.recycle()
                }
            }

        } finally {

            if (!alignedSource.bitmap.isRecycled) {
                alignedSource.bitmap.recycle()
            }

            if (!alignedTarget.bitmap.isRecycled) {
                alignedTarget.bitmap.recycle()
            }
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

        val canvas =
            Canvas(output)

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                    Paint.FILTER_BITMAP_FLAG
            )

        /*
         * Bitmap is created transparent.
         * Therefore only the transformed 128x128 face
         * contributes pixels; the rest remains transparent.
         */
        canvas.drawBitmap(
            alignedFace,
            matrix,
            paint
        )

        return output
    }

    fun close() {
        liae.close()
    }
}
