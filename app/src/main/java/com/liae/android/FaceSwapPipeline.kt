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
             * Convert aligned faces to:
             *
             * [1,128,128,3]
             * NHWC
             * BGR
             * 0..1
             */
            val sourceTensor =
                ImageTensor.bitmapToTensor(
                    alignedSource.bitmap
                )

            val targetTensor =
                ImageTensor.bitmapToTensor(
                    alignedTarget.bitmap
                )

            /*
             * Run LIAE.
             */
            val prediction =
                liae.run(
                    src = sourceTensor,
                    dst = targetTensor
                )

            lastDebugText =
                prediction.debugText

            /*
             * LIAE output:
             *
             * output_1:
             * [1,128,128,3]
             *
             * output_2:
             * [1,128,128,1]
             */
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
                 * Transform generated face from
                 * aligned 128x128 space back into
                 * target-image coordinates.
                 */
                val warpedFace =
                    warpToTarget(
                        alignedFace = swappedFace,
                        inverse = alignedTarget.inverse,
                        targetWidth = targetImage.width,
                        targetHeight = targetImage.height
                    )

                /*
                 * Transform the mask using exactly
                 * the same geometry.
                 */
                val warpedMask =
                    warpToTarget(
                        alignedFace = mask,
                        inverse = alignedTarget.inverse,
                        targetWidth = targetImage.width,
                        targetHeight = targetImage.height
                    )

                try {

                    /*
                     * Blend the warped face with
                     * the original target.
                     */
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

    /*
     * Warp a 128x128 aligned image back into
     * the full target-image coordinate system.
     *
     * The output starts fully transparent.
     *
     * This is important because pixels outside
     * the transformed face must NOT become part
     * of the final blend.
     */
    private fun warpToTarget(
        alignedFace: Bitmap,
        inverse: FloatArray,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {

        require(inverse.size == 6) {
            "Expected 6-value affine inverse matrix, " +
                "got ${inverse.size}"
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

        /*
         * Start completely transparent.
         */
        canvas.drawColor(
            Color.TRANSPARENT,
            PorterDuff.Mode.CLEAR
        )

        val paint =
            Paint(
                Paint.ANTI_ALIAS_FLAG or
                    Paint.FILTER_BITMAP_FLAG
            )

        paint.isFilterBitmap = true

        /*
         * Draw transformed aligned image.
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
