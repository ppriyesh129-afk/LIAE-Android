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

        /*
         * The exported LIAE checkpoint has the source identity
         * embedded in its Inter_AB weights.
         *
         * Therefore the ONNX graph itself takes only the
         * destination/target aligned face.
         *
         * We still detect the source face here so the existing
         * application flow remains compatible and to ensure
         * a valid source image was supplied.
         */
        val sourceFace =
            sourceFaces.maxByOrNull {
                it.score
            }
                ?: throw IllegalStateException(
                    "Source face selection failed"
                )

        /*
         * Keep sourceFace referenced intentionally.
         */
        @Suppress("UNUSED_VARIABLE")
        val verifiedSourceFace =
            sourceFace

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

        /*
         * Source alignment is retained for compatibility with
         * the existing pipeline, but the verified ONNX graph
         * does not consume sourceTensor.
         */
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
             * Convert target aligned face to:
             *
             * [1,128,128,3]
             * NHWC
             * BGR
             * 0..1
             */
            val targetTensor =
                ImageTensor.bitmapToTensor(
                    alignedTarget.bitmap
                )

            /*
             * Verified ONNX:
             *
             * INPUT
             * in_face [1,128,128,3]
             *
             * OUTPUT 0
             * output_1 [1,128,128,1]
             * destination mask
             *
             * OUTPUT 1
             * output_2 [1,128,128,3]
             * swapped face
             *
             * OUTPUT 2
             * output_3 [1,128,128,1]
             * source/swapped mask
             */
            val prediction =
                liae.run(
                    dst = targetTensor
                )

            lastDebugText =
                prediction.debugText

            val swappedFace =
                ImageTensor.tensorToBitmap(
                    prediction.rgb
                )

            /*
             * For the actual swapped face, use the
             * source/swapped-face mask (output_3).
             *
             * Destination mask is retained by the engine
             * for diagnostics and future refinement.
             */
            val mask =
                ImageTensor.maskToBitmap(
                    prediction.srcMask
                )

            try {

                val warpedFace =
                    warpToTarget(
                        alignedFace = swappedFace,
                        inverse = alignedTarget.inverse,
                        targetWidth = targetImage.width,
                        targetHeight = targetImage.height
                    )

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
